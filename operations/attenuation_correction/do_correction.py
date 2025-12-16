import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import tifffile
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

# GPU = True
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = int(sys.argv[5])
reference_z = int(sys.argv[6])
opening_radius = int(sys.argv[7])
mu_ref = float(sys.argv[8])
sigma_ref = float(sys.argv[9])
# NPY_DIR = sys.argv[10]


def correct_img(img):
    img_dtype = img.dtype
    info = np.iinfo(img_dtype)
    min_valid = info.min
    max_valid = info.max
    img = img.astype(np.float32)
    selem = disk(opening_radius)
    bg = ndi.grey_opening(img, footprint=selem)
    mu_z = float(bg.mean())
    sigma_z = float(bg.std())

    if sigma_ref is not None and sigma_z > 0:
        a = sigma_ref / sigma_z
    else:
        # Fall back to unity scaling if std is 0
        a = 1.0

    b = mu_ref - a * mu_z

    # Apply transform
    img_corr = a * img + b

    # Clip to valid range
    img_corr = np.clip(img_corr, min_valid, max_valid)

    # Cast back to original dtype
    img_corr = img_corr.astype(img_dtype)
    return img_corr


print("INPUT_DIR", INPUT_DIR)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)
# print("NPY_DIR", NPY_DIR)

files = sorted(glob(os.path.join(INPUT_DIR, "*.tif")))
input_file = files[z]
print("Input file:", input_file)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

img = tifffile.imread(input_file)
img = correct_img(img)
tifffile.imwrite(output_file, img)

# npy_file = os.path.join(
#     NPY_DIR,
#     f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.npy"
# )
# np.save(npy_file, np.array([img.min(), img.max()]))
print("Done")