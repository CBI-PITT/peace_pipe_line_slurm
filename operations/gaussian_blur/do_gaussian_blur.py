import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import gaussian

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def gaussian_blur(image, sigma):
    image = gaussian(image, sigma=sigma, preserve_range=True)
    image = np.clip(image, 0, 65535).astype(np.uint16)
    return image


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = sys.argv[5]
sigma = float(sys.argv[6])

print("INPUT_DIR", INPUT_DIR)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)
print("sigma", sigma)

input_file = os.path.join(
    INPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

print("Running gaussian blur")
img = tifffile.imread(input_file)
img = gaussian_blur(img, sigma)
tifffile.imwrite(output_file, img)
print("Done")
