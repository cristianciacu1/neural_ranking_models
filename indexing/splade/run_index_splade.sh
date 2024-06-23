#!/bin/bash

#SBATCH --job-name="deep"
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1
module load python
module load py-pip
module load openjdk
module load cuda

# Conda environment
conda create -n py39 python=3.9
conda activate py39

# Install an older version of rust (required for the compatibility
# with tokenizers)
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
rustup install nightly-2022-01-01
rustup default 1.59.0

# Install dependencies
python -m pip install --user tokenizers==0.11.6 python-terrier==0.10.0 fast-forward-indexes==0.2.0 jupyter ipywidgets
pip install -q git+https://github.com/naver/splade.git
pip install -q git+https://github.com/cmacdonald/pyt_splade.git

# Run the experiment
srun time python index_splade.py

# Exit the environment
conda deactivate
