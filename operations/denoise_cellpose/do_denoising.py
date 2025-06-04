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

os.umask(settings.UMASK)

GPU = True
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = sys.argv[5]
MODEL = sys.argv[6]
DIAMETER = int(sys.argv[7])
stack_min = int(sys.argv[8])
stack_max = int(sys.argv[9])


def denoise_img(img):
    dn = denoise.DenoiseModel(model_type=MODEL, gpu=GPU)
    img = img.astype('float32')
    img = (img - stack_min)/(stack_max - stack_min)
    img_dn = dn.eval(img, channels=None, diameter=DIAMETER)
    img_dn = np.squeeze(img_dn)
    return img_dn


print("INPUT_DIR", INPUT_DIR)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)

input_file = os.path.join(
    INPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

print("Stretching contrast")
img = tifffile.imread(input_file)
img = denoise_img(img)
tifffile.imwrite(output_file, img)
print("Done")