#!/bin/bash

#SBATCH --job-name="di_scifact"
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load miniconda3

# Create conda environment
conda activate sprint_env

# Dataset
DATASET="scifact"
TOPIC_SPLIT_1="test"
TOPIC_SPLIT_2="train"
QUERIES_PER_DOC=3

# 0. Generate queries
srun time -v python generate_queries.py -d $DATASET -q $QUERIES_PER_DOC

# 1. Convert queries to the correct format
srun time -v python convert_queries.py -d $DATASET -q $QUERIES_PER_DOC

# 2. Augment the documents
chmod +x preprocessing.sh
./preprocessing.sh $DATASET $QUERIES_PER_DOC

# 3. Encode
chmod +x encode.sh
./encode.sh $DATASET

# 4. Quantize
chmod +x quantize.sh
./quantize.sh $DATASET

# 5. Index
chmod +x index.sh
./index.sh $DATASET

# 6. Reformat Queries
chmod +x reformat_query.sh
./reformat_query.sh $DATASET $TOPIC_SPLIT_1
./reformat_query.sh $DATASET $TOPIC_SPLIT_2

# 7. Search
chmod +x search.sh
./search.sh $DATASET $TOPIC_SPLIT_1
./search.sh $DATASET $TOPIC_SPLIT_2