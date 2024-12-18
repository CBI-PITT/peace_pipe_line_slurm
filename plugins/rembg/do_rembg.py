# conda activate rembg
import os
print("COMPUTER", os.uname().nodename)
import sys

import numpy as np
import tifffile


def run_rembg(image):
    from rembg import remove
    output = remove(image)
    fg = output[:, :, 0] > 0
    image = image * fg
    return image


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = sys.argv[5]

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

print("Removing background")
img = tifffile.imread(input_file)
img = run_rembg(img)
tifffile.imwrite(output_file, img)
print("Done")
