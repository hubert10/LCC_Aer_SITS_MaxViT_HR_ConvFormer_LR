import torch
import time
from torch import nn
import numpy as np
import numpy as np
from scipy import stats
import torchvision.transforms as T
from src.utils import functions
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision.transforms import functional as TFF


class SegmentationTask(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        criterion,
        optimizer,
        config,
        scheduler=None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.CM = None
        self.init_aux_vars()
        self.sits_aux_loss_main_weight = self.config["hyperparams"]["sits_aux_loss_main_weight"]
        self.aux_loss_weight = self.config["hyperparams"]["aux_loss_weight"]
        self.feature_maps_downsample_factor = 49

    # CLASS WEIGHTS UPDATING DURING TRAINING

    def init_aux_vars(self):
        """Initialization of variables used in the training process."""
        self.CM = np.zeros(
            [self.config["inputs"]["num_classes"], self.config["inputs"]["num_classes"]], np.float32
        )

    def downsample_label_map_majority_vote_with_crop(
        self, labels, original_size=512, cropped_size=500, output_size=10
    ):
        """
        Downsamples multi-class label maps using majority vote after cropping.

        Args:
            labels (torch.Tensor): Input label maps of shape [N, 512, 512].
            original_size (int): Original spatial size (assumed square). Default is 512.
            cropped_size (int): Desired spatial size after cropping (assumed square). Default is 500.
            output_size (int): Desired output spatial size (assumed square). Default is 10.

        Returns:
            torch.Tensor: Downsampled label maps of shape [N, 10, 10].
        """
        
        N, H, W = labels.shape
        assert (
            H == original_size and W == original_size
        ), f"Input label maps must be of shape [N, {original_size}, {original_size}]"
        assert (
            cropped_size % output_size == 0
        ), f"cropped_size must be divisible by output_size. Got cropped_size={cropped_size}, output_size={output_size}"

        # Step 1: Crop the label maps to [N, 500, 500]
        # Assuming center crop: remove 6 pixels from each side
        crop_margin = (original_size - cropped_size) // 2  # 6 pixels
        labels_cropped = labels[
            :,
            crop_margin : crop_margin + cropped_size,
            crop_margin : crop_margin + cropped_size,
        ]

        # Step 2: Reshape to [N, output_size, block_size, output_size, block_size]
        block_size = cropped_size // output_size  # 50
        labels_reshaped = labels_cropped.view(
            N, output_size, block_size, output_size, block_size
        )

        # Step 3: Permute to [N, output_size, output_size, block_size, block_size]
        labels_permuted = labels_reshaped.permute(0, 1, 3, 2, 4)

        # Step 4: Flatten the block pixels to [N, output_size, output_size, block_size * block_size]
        labels_flat = labels_permuted.reshape(
            N, output_size, output_size, block_size * block_size
        )

        # Step 5: Compute mode along the last dimension (majority vote)
        mode, _ = torch.mode(labels_flat, dim=-1)

        return mode  # [N, 10, 10]

    def setup(self, stage=None):
        if stage == "fit":
            self.train_epoch_loss, self.val_epoch_loss = None, None
            self.train_epoch_metrics, self.val_epoch_metrics = None, None
            # Intersection over union or Jaccard index calculation
            # self.train_metrics = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')

            self.train_metrics = MetricCollection(
                {"val/miou": MulticlassJaccardIndex(self.num_classes, average="macro")}
            )
            # macro: Calculate statistics for each label and average them
            # self.val_metrics = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')

            # init metrics for evaluation
            self.val_metrics = MetricCollection(
                {"val/miou": MulticlassJaccardIndex(self.num_classes, average="macro")}
            )
            self.train_loss = MeanMetric()
            self.val_loss = MeanMetric()

        elif stage == "validate":
            self.val_epoch_loss, self.val_epoch_metrics = None, None
            # self.val_metrics = MulticlassJaccardIndex(num_classes=self.num_classes, average='macro')

            self.val_metrics = MetricCollection(
                {"val/miou": MulticlassJaccardIndex(self.num_classes, average="macro")}
            )
            self.val_loss = MeanMetric()

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

    # modified
    def forward(self, input_patch, input_spatch, input_dates):
        _, _, logits = self.model(input_patch, input_spatch, input_dates)
        return logits

    def step(self, batch):
        patch = batch["patch"]
        patch_spot6 = batch["s6_patch"]
        spatch = batch["spatch"]
        dates = batch["dates"]
        targets = batch["labels"]#.long()
        targets_sp = batch["labels"]#.long()

        (
            logits_sits,
            multi_lvls_outs,
            logits_aerial,
        ) = self.model(patch, spatch, dates)

        aerial_criterion = self.criterion[0]
        sits_criterion = self.criterion[1]

        # aerial_criterion.weight = self.class_weights.to(logits_aerial.device)
        # sits_criterion.weight = self.class_weights.to(logits_sits.device)
        
        targets = targets_sp = torch.argmax(targets, dim=1) if targets.ndim == 4 else targets
        loss_aerial = aerial_criterion(logits_aerial, targets)  # aerial images

        # Down-sample the GT labels from the Aerial Image
        # by a factor of 1/50 to match the GSD of the SITS
        # branch i.e SITS(10m): 0.2m x 50

        target_aux = self.downsample_label_map_majority_vote_with_crop(
            targets_sp
        ).long()

        # The supervision of the SITS branch is done at 1om GSD
        # The SITS branch is trained with the 10m GSD features
        # The main sits loss is computed comparing the combined and upsampled sits features
        # at the 10m GSD with the downsampled GT labels from 20cm to 10m

        transform = T.CenterCrop((10, 10))
        main_loss_sits = sits_criterion(transform(logits_sits), target_aux)  # satellite images

        # Auxiliary losses
        aux_loss2 = sits_criterion(transform(multi_lvls_outs[2]), target_aux)
        aux_loss3 = sits_criterion(transform(multi_lvls_outs[1]), target_aux)
        aux_loss4 = sits_criterion(transform(multi_lvls_outs[0]), target_aux)

        loss_sits = self.sits_aux_loss_main_weight * main_loss_sits + (
            self.aux_loss_weight * aux_loss2
            + self.aux_loss_weight * aux_loss3
            + self.aux_loss_weight * aux_loss4
        )

        loss = (
            loss_sits * self.config["hyperparams"]["w_aerial_sits"][0]
            + loss_aerial * self.config["hyperparams"]["w_aerial_sits"][1]
        )  # loss is the sum of aerial and satellite losses

        with torch.no_grad():
            preds = logits_aerial.argmax(dim=1)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.train_loss.update(loss)
        self.train_metrics(preds=preds, target=targets)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.config["tasks"]["sync_dist"],
        )
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        print(f"Time for epoch {self.current_epoch}: {time.time() - self.start_time}")
        self.train_epoch_loss = self.train_loss.compute()
        self.train_epoch_metrics = self.train_metrics.compute()

        self.log(
            "train_loss",
            self.train_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.config["tasks"]["sync_dist"],
            rank_zero_only=True,
        )

        self.train_loss.reset()
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        self.val_loss.update(loss)
        self.val_metrics(preds=preds, target=targets)

        # # UPDATE CLASS WEIGHTS
        # if self.config["models"]["t_convformer"]["update_weights"] == True:
        #     self.evaluate_training(preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        self.val_epoch_loss = self.val_loss.compute()
        self.val_epoch_metrics = self.val_metrics.compute()
        self.log(
            "val_loss",
            self.val_epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.config["tasks"]["sync_dist"],
            rank_zero_only=True,
        )
        self.log(
            "val_miou",
            self.val_epoch_metrics["val/miou"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.config["tasks"]["sync_dist"],
            rank_zero_only=True,
        )
        self.log_dict(self.val_epoch_metrics, on_epoch=True)
        self.val_loss.reset()
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.config["hyperparams"]["use_tta"] == True:
            # logits_aerial = self.forward(
            #     self.config, batch["patch"], batch["spatch"], batch["dates"], batch["mtd"]
            # )
            logits_aerial = self.forward(
                batch["patch"], batch["spatch"], batch["dates"]
            )
            logits_aerial_tta = TFF.hflip(
                #     self.forward(
                #         self.config,
                #         TFF.hflip(batch["patch"]),
                #         TFF.hflip(batch["spatch"]),
                #         batch["dates"],
                #         batch["mtd"],
                #     )
                # )
                self.forward(
                    TFF.hflip(batch["patch"]),
                    TFF.hflip(batch["spatch"]),
                    batch["dates"],
                )
            )

            logits = logits_aerial + logits_aerial_tta
        else:
            # logits_aerial = self.forward(
            #     self.config, batch["patch"], batch["spatch"], batch["dates"], batch["mtd"]
            # )

            logits_aerial = self.forward(
                batch["patch"], batch["spatch"], batch["dates"]
            )
        logits = logits_aerial

        proba = torch.softmax(logits, dim=1)
        batch["preds"] = torch.argmax(proba, dim=1)
        return batch

    def configure_optimizers(self):
        if self.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "strict": True,
                "name": "Scheduler",
            }
            config_ = {"optimizer": self.optimizer, "lr_scheduler": lr_scheduler_config}
            return config_
        else:
            return self.optimizer
