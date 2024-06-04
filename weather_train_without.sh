#!/bin/bash  
#SBATCH --cpus-per-task=1   
#SBATCH --gres=gpu:1  
#SBATCH --mem=16G  
#SBATCH --time=0-1:00:00  
#SBATCH --output=test-%j.out


# launch the train file
python temperature_train.py
