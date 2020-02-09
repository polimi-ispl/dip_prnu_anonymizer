#!/bin/bash

#### Select the number of nodes to be used (1 node = 8 cores)
#PBS -l select=1:ncpus=16:mem=20gb:ngpus=1

#### Job name
#PBS -N DIP_g6

#### File di output e di errore
#PBS -o dip_g6.out
#PBS -e dip_g6.stderr

#### Submission queue
#PBS -q xagigpu

####  Activate environment

source activate dip
cd /gpfs/scratch/usera06ptm/a06ptm04/fpicetti/dip_prnu_anonymizer

python main.py --device Nikon_D200_1 --pics_idx 0 20 --outpath gamma0.1 --gamma .1
python main.py --device Nikon_D200_1 --pics_idx 0 20 --outpath gamma1 --gamma 1
python main.py --device Nikon_D200_1 --pics_idx 0 20 --outpath gamma0.001 --gamma 1e-3

python /galileo/home/usera06ptm/a06ptm04/fpicetti/slack.py  -m "Finished gamma of Nikon_D200_1" -u francesco.picetti sara.mandelli paolo.bestagini vincenzo.lipari
