#!/bin/bash  # The shebang line specifies the script interpreter.
#SBATCH --cpus-per-task=1   # Allocate 1 CPU core per task.
#SBATCH --gres=gpu:1  # Request 1 GPU for the job.
#SBATCH --mem=16G  # Memory per node
#SBATCH --time=0-1:00:00  # Time (DD-HH:MM)
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=your_email_address
#SBATCH --mail-type=ALL

#load modules
module load python/3.11.5 scipy-stack

#create and activate virtual environment
virtualenv --no-download ENV
source ENV/bin/activate

#install Packages
pip install --no-index --upgrade pip
pip install --no-index scikit_learn==1.3.1
pip install --no-index seaborn==0.13.2


# launch the train file
python temperature_train.py
