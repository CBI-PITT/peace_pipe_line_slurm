import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage import exposure

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)


def stretch_contrast(image):
    image = image.astype(np.float32)
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    image = (image * 65535).astype(np.uint16)
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
    # f'resolution_level_{resolution_level}',
    # f'channel_{channel}',
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_file = os.path.join(
    OUTPUT_DIR,
    # 'stretch_contrast',
    # f"r{str(resolution_level)}_c{str(channel)}_contrast_stretched",
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

print("Stretching contrast")
img = tifffile.imread(input_file)
img = stretch_contrast(img)
tifffile.imwrite(output_file, img)
print("Done")
