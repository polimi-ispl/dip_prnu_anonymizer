# DIPPAS: A Deep Image Prior PRNU Anonymization Scheme
<img src="assets/dip_scheme.png" width="500">


This is the official repository of **DIPPAS: A Deep Image Prior PRNU Anonymization Scheme**,
submitted to IEEE Transaction on Information Forensics and Security and currently available on [arXiv](https://arxiv.org/pdf/2012.03581.pdf).

## Code

### Prerequisites

- Install conda
- Create the `dippas` environment with `environment.yml`
```bash
$ conda env create -f environment.yml
$ conda activate dippas
```
- The code works fine with Nvidia Tesla V100 GPUs

### Train

Run `main_1.py` ... @Fra vuoi spiegare qui i passaggi necessari?

```bash
$ python3 train_cnn.py --model_dir ./models/Pcn_crop224 --crop_size 224 --base_network Pcn
```

## Credits
[ISPL: Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)
- Francesco Picetti (francesco.picetti@polimi.it)
- Sara Mandelli (sara.mandelli@polimi.it)
- Paolo Bestagini (paolo.bestagini@polimi.it) 
