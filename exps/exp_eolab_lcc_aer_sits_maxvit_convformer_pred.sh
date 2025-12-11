#!/bin/bash 
#SBATCH --job-name=lcc_aer_sits_maxvit_hr_convformer_lr_pred
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=20:00:00
#SBATCH --mail-user=kanyamahanga@ipi.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output logs/lcc_aer_sits_maxvit_hr_convformer_lr_pred_%j.out
#SBATCH --error logs/lcc_aer_sits_maxvit_hr_convformer_lr_pred_%j.err
export CONDA_ENVS_PATH=$HOME/.conda/envs
DATA_DIR="/my_data/"
export DATA_DIR
source /home/eouser/flair_venv/bin/activate
which python
cd $HOME/exp_2026/LCC_Aer_SITS_MaxViT_HR_ConvFormer_LR
srun python main.py --config_file=./configs/train_main/





