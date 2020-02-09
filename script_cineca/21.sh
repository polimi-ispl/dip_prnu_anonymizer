#!/bin/bash

#### Select the number of nodes to be used (1 node = 8 cores)
#PBS -l select=1:ncpus=16:mem=20gb:ngpus=1

#### Job name
#PBS -N DIP_21

#### File di output e di errore
#PBS -o dip_21.out
#PBS -e dip_21.stderr

#### Submission queue
#PBS -q xagigpu

####  Activate environment

source activate dip
cd /gpfs/scratch/usera06ptm/a06ptm04/fpicetti/dip_prnu_anonymizer

python main.py --device Nikon_D70_1 --pics_idx 0  33   --outpath default
python /galileo/home/usera06ptm/a06ptm04/fpicetti/slack.py  -m "Finished first third of Nikon_D70_1" -u francesco.picetti sara.mandelli paolo.bestagini vincenzo.lipari
