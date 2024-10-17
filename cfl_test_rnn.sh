#!/bin/bash -l
#SBATCH -o cfl_rnn_output.txt          # Standard output file
#SBATCH -e cfl_rnn_error.txt           # Error output file
#SBATCH -p general                     # Partition to run on
#SBATCH --cpus-per-task=4              # Number of CPUs
#SBATCH --mem=16GB                     # Memory allocation
#SBATCH --gpus=1                       # Number of GPUs
#SBATCH --mail-user=mannan2@usf.edu    # Notifications
#SBATCH --mail-type=BEGIN,END,FAIL     # When to notify

# Load conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cfl_test_env

# Verify active environment
echo "Active Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Running the Python script
python cfl_test_rnn.py