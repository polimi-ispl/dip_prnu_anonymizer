#!/bin/bash
img_list=`ls /gpfs/scratch/usera06ptm/a06ptm04/fpicetti/dip_prnu_anonymizer/results/exit_siVGG_siREG_jpg/*.hdf5`
for i in $img_list;
do
python extract_ncc_blocks.py --input_path $i
done
# modify the img_list accordingly to the desired folder