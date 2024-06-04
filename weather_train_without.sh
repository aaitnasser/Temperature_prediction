#!/bin/bash  # The shebang line specifies the script interpreter.
#SBATCH --cpus-per-task=1   # Allocate 1 CPU core per task.
#SBATCH --gres=gpu:1  # Request 1 GPU for the job.
#SBATCH --mem=16G  # Memory per node
#SBATCH --time=0-1:00:00  # Time (DD-HH:MM)
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=your_email_address
#SBATCH --mail-type=ALL

# launch the train file
python temperature_train.py
