#!/bin/bash

#### Select the number of nodes to be used
#PBS -l select=1:ncpus=16:mem=120gb:ngpus=1

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

# the one that seems to work better
python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 0 0 4 4
# varying input depth
python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 0 0 4 4 --input_depth 1024
# varying noise dist
python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 0 0 4 4 --noise_dist normal
# varying the network
python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 128 128 128 128

# varying alpha
# --alpha 0.00
# --alpha 0.01
# --alpha 0.10
# --alpha 1.00


# # as done by Fantong
# python main_single.py --filters 128 128 128 128 --skip 128 128 128 128 --epochs 2000 --input_depth 1024 --activation LeakyReLU --upsample bilinear
#
# # as proposed by Fantong
# python main_single.py --filters 256 256 256 256 --skip 128 128 128 128
#
# # using also DnCNN (add NCC between PRNU and DnCNN(output))
# --beta 0.01

python /galileo/home/usera06ptm/a06ptm04/fpicetti/slack.py -u francesco.picetti
