#!/bin/bash -l
#SBATCH -o cflang_output.txt          # Standard output file
#SBATCH -e cflang_error.txt           # Error output file
#SBATCH -p general                     # Partition to run on
#SBATCH --cpus-per-task=4              # Number of CPUs
#SBATCH --mem=16GB                     # Memory allocation
#SBATCH --gpus=1                       # Number of GPUs
#SBATCH --mail-user=mannan2@usf.edu    # Notifications
#SBATCH --mail-type=BEGIN,END,FAIL     # When to notify

# Activating the conda environment
source activate cfl_test_env

# Running the Python script
python Cfl_test.py
