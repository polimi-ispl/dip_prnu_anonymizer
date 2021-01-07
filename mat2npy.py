import numpy as np
import os
from glob import glob
from scipy.io import loadmat

ROOT = "/nas/home/fpicetti/dip_prnu_anonymizer/dresdenJPG"

for f in glob(os.path.join(ROOT, "*/*.mat")):
    name, _ = os.path.splitext(f)
    if name.split("/")[-1] == "prnu":
        prnu = loadmat(f)['prnu']
        np.save(name+'.npy', prnu)
        print(f)
        os.remove(f)
    else:
        os.remove(f)
        print("Removed ", f)
