#!/bin/bash

#SBATCH --job-name="di_nf"
#SBATCH --time=00:45:00
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load miniconda3

# Create conda environment
conda activate sprint_env

DATASET_NAME='nfcorpus'
TOPIC_SPLIT='test'

srun python -m sprint_toolkit.inference.aio \
    --encoder_name unicoil \
    --ckpt_name castorini/unicoil-noexp-msmarco-passage \
    --data_dir /scratch/cciacu/test_dir/indexing/unicoil/datasets/$DATASET_NAME \
    --data_name beir_$DATASET_NAME \
    --gpus 0 \
    --output_dir /scratch/cciacu/test_dir/indexing/unicoil/runs/$DATASET_NAME/$TOPIC_SPLIT \
    --do_quantization \
    --quantization_method range-nbits \
    --original_score_range 5 \
    --quantization_nbits 8 \
    --original_query_format beir \
    --topic_split $TOPIC_SPLIT 