import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import albumentations as A
from src.data.utils_data.paths import get_datasets
from src.models.sits_aerial_seg_model import SITSAerialSegmenter
from src.models.task_module import SegmentationTask
from src.tasks.module_setup import build_data_module

from src.utils.utils_prints import (
    print_config,
    print_recap,
    print_metrics,
    print_inference_time,
    print_iou_metrics,
    print_f1_metrics,
    print_overall_accuracy,
)

from src.utils.utils_dataset import read_config
from src.utils.prediction_writer import PredictionWriter

torch.set_float32_matmul_precision("medium")
from src.utils.metrics import generate_miou, generate_mf1s, generate_metrics

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

argParser = argparse.ArgumentParser()
argParser.add_argument("--config_file", help="Path to the .yml config file")


def main(config):
    seed_everything(2022, workers=True)
    out_dir = Path(config["paths"]["out_folder"], config["paths"]["out_model_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    d_train, d_val, d_test = get_datasets(config)

    print_recap(config, d_train, d_val, d_test)

    # Get LightningDataModule
    data_module = build_data_module(config, dict_train=d_train, dict_val=d_val, dict_test=d_test)

    # data_module = FlairDataModule(
    #     config=config,
    #     dict_train=d_train,
    #     dict_val=d_val,
    #     dict_test=d_test,
    #     batch_size=config['hyperparams']["batch_size"],
    #     num_workers=config['hardware']["num_workers"],
    #     drop_last=True,
    #     use_augmentations=config['modalities']['pre_processings']["use_augmentation"],
    # )

    import ssl

    # Fixing URLError when loading pre-trained weights!
    ssl._create_default_https_context = ssl._create_unverified_context
    model = SITSAerialSegmenter(config)

    # Optimizer and Loss
    if config["hyperparams"]["optimizer"] == "sdg":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=float(config["models"]["hyperparams"]["lr"])
        )  # weight_decay=self.C.wdec,

    elif config["hyperparams"]["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["hyperparams"]["lr"]))

    elif config["hyperparams"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["hyperparams"]["lr"]),
            betas=(float(config["hyperparams"]["beta_1"]), float(config["hyperparams"]["beta_2"])),
            weight_decay=float(config["hyperparams"]["w_dec"]),
        )
    else:
        print("no implemented optimizer chosen!")

    # Reduce learning rate when a metric has stopped improving.
    # Models often benefit from reducing the learning rate by a factor
    # of 2-10 once learning stagnates. This scheduler reads a metrics
    # quantity and if no improvement is seen for a patience number
    # of epochs, the learning rate is reduced.

    if config["hyperparams"]["scheduler"] == "StepLR":
        scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.7)
    elif config["hyperparams"]["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=5,  # 10
            cooldown=4,
            min_lr=1e-7,
        )
    else:
        scheduler = None
    # changed to use only one vector for thr weights!

    criterion_vhr = nn.CrossEntropyLoss()
    criterion_hr = nn.CrossEntropyLoss()

    if scheduler is not None:
        seg_module = SegmentationTask(
            model=model,
            num_classes=config["inputs"]["num_classes"],
            criterion=nn.ModuleList([criterion_vhr, criterion_hr]),
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
        )
    else:
        seg_module = SegmentationTask(
            model=model,
            num_classes=config["inputs"]["num_classes"],
            criterion=nn.ModuleList([criterion_vhr, criterion_hr]),
            optimizer=optimizer,
            config=config,
        )
    # Callbacks

    ckpt_callback = ModelCheckpoint(
        monitor=config["saving"]["ckpt_monitor"],
        dirpath=os.path.join(out_dir, "checkpoints"),
        filename="ckpt-{epoch:02d}-{val_loss:.2f}-{val_miou:.2f}"
        + "_"
        + config["paths"]["out_model_name"],
        save_top_k=1,
        mode=config["saving"]["ckpt_monitor_mode"],
        save_last=True,
        # save_weights_only=True,  # can be changed accordingly
    )

    early_stop_callback = EarlyStopping(
        monitor=config["saving"]["ckpt_monitor"],
        min_delta=0.00,
        patience=20,  # if no improvement after 20 epoch, stop learning.
        mode=config["saving"]["ckpt_monitor_mode"],
    )

    prog_rate = TQDMProgressBar(refresh_rate=config["saving"]["progress_rate"])

    callbacks = [
        ckpt_callback,
        early_stop_callback,
        prog_rate,
    ]

    # Logger

    logger = TensorBoardLogger(
        save_dir=out_dir,
        name=Path(
            "tensorboard_logs" + "_" + config["paths"]["out_model_name"]
        ).as_posix(),
    )

    loggers = [logger]

    # Define trainer and run

    if config["hardware"]["strategy"] == "ddp":
        config["hardware"]["strategy"] = DDPStrategy(find_unused_parameters=True)

    # Define trainer and run

    trainer = Trainer(
        detect_anomaly=False,
        accelerator=config["hardware"]["accelerator"],
        devices=config["hardware"]["gpus_per_node"],
        strategy=config["hardware"]["strategy"],
        num_nodes=config["hardware"]["num_nodes"],
        max_epochs=config["hyperparams"]["num_epochs"],
        log_every_n_steps=config["saving"]["log_every_n_steps"],  # 50 default used by the Trainer
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=config["saving"]["enable_progress_bar"],
    )

    if config["saving"]["mode"] == "train":
        if config["saving"]["resume"]:
            ckpt_path = Path(
                config["paths"]["out_folder"],
                config["paths"]["out_model_name"],
                "checkpoints/last.ckpt",
            )
            trainer.fit(seg_module, datamodule=data_module, ckpt_path=ckpt_path)
        else:
            trainer.fit(seg_module, datamodule=data_module)

        ## Check metrics on validation set

        trainer.validate(seg_module, datamodule=data_module)

    else:
        # Predict
        writer_callback = PredictionWriter(
            output_dir=os.path.join(
                out_dir, "predictions" + "_" + config["paths"]["out_model_name"]
            ),
            write_interval="batch",
        )

        # Predict Trainer
        trainer = Trainer(
            accelerator=config["hardware"]["accelerator"],
            devices=config["hardware"]["gpus_per_node"],
            strategy=config["hardware"]["strategy"],
            num_nodes=config["hardware"]["num_nodes"],
            callbacks=[writer_callback],
            enable_progress_bar=config["saving"]["enable_progress_bar"],
        )

        def get_best_model(model_path):
            for file in os.listdir(model_path):
                if file.startswith("ckpt"):
                    return file
            # return None

        model_path = Path(
            config["paths"]["out_folder"],
            config["paths"]["out_model_name"],
            "checkpoints",
        )

        ckpt_path = os.path.join(model_path, get_best_model(model_path))
        ## Enable time measurement 
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        #     enable_timing=True
        # )
        
        # starter.record()
        
        trainer.predict(
            seg_module,
            datamodule=data_module,
            return_predictions=False,
            ckpt_path=ckpt_path,
        )
    
        # ender.record()
        # torch.cuda.synchronize()
    
        # inference_time_seconds = starter.elapsed_time(ender) / 1000.0
        # print_inference_time(inference_time_seconds, config)

        # Compute mIoU over the predictions - not done here as the test labels are not available, but if needed, you can use the generate_miou function from metrics.py
        csv_path = Path(config["paths"]["test_csv"])
        df = pd.read_csv(csv_path, sep=";")
        truth_msk = df[config["labels"][0]].tolist()

        pred_msk = os.path.join(
            out_dir, "predictions" + "_" + config["paths"]["out_model_name"]
        )
        # mIou, ious = generate_miou(config, truth_msk, pred_msk)
        # mf1, f1s, oa = generate_mf1s(truth_msk, pred_msk)

        # print_iou_metrics(mIou, ious)
        # print_f1_metrics(mf1, f1s)
        # print_overall_accuracy(oa)
        
        output_dir=os.path.join(
                out_dir, "predictions" + "_" + config["paths"]["out_model_name"]
            )
        generate_metrics(config, truth_msk, pred_msk, output_dir, config["labels"][0])


    @rank_zero_only
    def print_finish():
        print("--  [FINISHED.]  --", f"output dir : {out_dir}", sep="\n")

    print_finish()


if __name__ == "__main__":
    args = argParser.parse_args()

    config = read_config(args.config_file)

    # printing model configuration
    print_config(config)

    main(config)
