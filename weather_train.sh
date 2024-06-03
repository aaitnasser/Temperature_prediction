#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-1:00:00
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=your_email_address
#SBATCH --mail-type=ALL

echo "Current working directory: `pwd`"
echo "Starting run at: `date`"

#load modules
module load python/3.11.5 scipy-stack

#create and activate virtual environment
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index scikit_learn==1.3.1
pip install --no-index seaborn==0.13.2
#python --version


python temperature_train.py
