#!/bin/bash
####  Activate environment
conda activate dip
cd /nas/home/fpicetti/dip_prnu_anonymizer

# the one that seems to work better
#python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 0 0 4 4 --gamma 0.01
# varying input depth
#python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 0 0 4 4 --gamma 0.01 --input_depth 1024
# varying noise dist
#python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 0 0 4 4 --gamma 0.01 --noise_dist normal
# varying the network
#python main.py --pics_per_dev 10 --nccd --save_every 50 --filters 128 128 128 128 --skip 128 128 128 128 --gamma 0.01

# new runs
python main.py --device Nikon_D70_0 --pics_per_dev -1 --save_png_every 50 --outpath default_allpics


# # as done by Fantong
# python main_single.py --filters 128 128 128 128 --skip 128 128 128 128 --epochs 2000 --input_depth 1024 --activation LeakyReLU --upsample bilinear
#
# # as proposed by Fantong
# python main_single.py --filters 256 256 256 256 --skip 128 128 128 128
#
# # using also DnCNN (add NCC between PRNU and DnCNN(output))
# --beta 0.01

python /nas/home/fpicetti/slack.py -u francesco.picetti sara.mandelli
