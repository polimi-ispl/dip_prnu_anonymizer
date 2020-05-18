#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/exit_noVGG_siREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/exit_noVGG_noREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/exit_siVGG_siREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/exit_siVGG_noREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/redx_noVGG_siREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/redx_noVGG_noREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/redx_siVGG_siREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done

#!/bin/bash
img_list=`ls /nas/home/fpicetti/dip_prnu_anonymizer/results/redx_siVGG_noREG/*.hdf5`
for i in $img_list;
do
python extract_ncc_bash.py --input_path $i
done
