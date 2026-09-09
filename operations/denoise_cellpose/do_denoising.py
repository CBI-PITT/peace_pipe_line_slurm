import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from cellpose import denoise

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.z_range import resolve_tiff_path

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")

GPU = True
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = int(sys.argv[5])
MODEL = sys.argv[6]
DIAMETER = int(sys.argv[7])
stack_min = float(sys.argv[8])
stack_max = float(sys.argv[9])
NPY_DIR = sys.argv[10]


def denoise_img(img):
    dn = denoise.DenoiseModel(model_type=MODEL, gpu=GPU)
    img = img.astype('float32')
    if stack_max == stack_min:
        img = np.zeros_like(img, dtype=np.float32)
    else:
        img = (img - stack_min)/(stack_max - stack_min)
    img_dn = dn.eval(img, channels=None, diameter=DIAMETER)
    img_dn = np.squeeze(img_dn)
    return img_dn


print("INPUT_DIR", INPUT_DIR)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)
print("NPY_DIR", NPY_DIR)


def resolve_input_file():
    metadata = None
    metadata_path = os.path.join(INPUT_DIR, f'.{settings.INFO_FILE_NAME}')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    return resolve_tiff_path(
        INPUT_DIR,
        metadata,
        resolution_level,
        channel,
        z,
    )


input_file = resolve_input_file()
print("Input file:", input_file)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{int(resolution_level):02d}_t00_c{int(channel):02d}_z{z:04d}.tif"
)

print("Stretching contrast")
img = tifffile.imread(input_file)
img = denoise_img(img)
tifffile.imwrite(output_file, img)

npy_file = os.path.join(
    NPY_DIR,
    f"r{int(resolution_level):02d}_t00_c{int(channel):02d}_z{z:04d}.npy"
)
np.save(npy_file, np.array([img.min(), img.max()]))
print("Done")
