#!/bin/sh

#SBATCH --job-name="dct_fever"
#SBATCH --time=17:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=90G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load cuda
module load openjdk
module load miniconda3

export IR_DATASETS_HOME=/scratch/cciacu/.ir_datasets

# Create conda environment
conda activate deepct_py39

# Run the experiment
srun time python index.py