#!/bin/bash

#### Select the number of nodes to be used
#PBS -l select=1:ncpus=16:mem=20gb:ngpus=1

#### Job name
#PBS -N dip

#### File di output e di errore
#PBS -o dip_prnu.out
#PBS -e dip_prnu.stderr

#### Submission queue
#PBS -q xagigpu

####  Activate environment
conda activate dip
cd /gpfs/scratch/usera06ptm/a06ptm04/fpicetti/dip_prnu_anonymizer

python main.py --device Nikon_D70_0 --pics_per_dev 10 --save_png_every 5

python /galileo/home/usera06ptm/a06ptm04/fpicetti/slack.py -u francesco.picetti
