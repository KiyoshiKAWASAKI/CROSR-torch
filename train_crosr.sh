#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -e errors/
#$ -N crosr_seed_3

# Required modules
module load conda
conda init bash
source activate new_msd_net

python train_net.py