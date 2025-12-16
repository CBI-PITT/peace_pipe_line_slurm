import os
import sys
from glob import glob
from pathlib import Path

import tifffile
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk


this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
reference_z = int(sys.argv[5])
opening_radius = int(sys.argv[6])

files = sorted(glob(os.path.join(INPUT_DIR, "*.tif")))
ref_file = files[reference_z]
print("Reference file:", ref_file)

output_file = os.path.join(OUTPUT_DIR, f"stats_reference_z_{reference_z}.npy")

if not os.path.exists(output_file):
    img = tifffile.imread(ref_file)
    img = img.astype(np.float32)
    selem = disk(opening_radius)
    ref_bg = ndi.grey_opening(img, footprint=selem)
    mu_ref = float(ref_bg.mean())
    sigma_ref = float(ref_bg.std())
    np.save(output_file, np.array([mu_ref, sigma_ref]))
