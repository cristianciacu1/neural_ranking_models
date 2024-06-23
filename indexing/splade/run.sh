#!/bin/bash

#SBATCH --job-name="spl_fever"
#SBATCH --time=17:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load openjdk
module load cuda
module load miniconda3

export IR_DATASETS_HOME=/scratch/cciacu/.ir_datasets/
export IR_DATASETS_TMP=/scratch/cciacu/tmp/.ir_datasets/

# Conda environment
conda activate rp-splade

# Run the experiment
srun python index.py