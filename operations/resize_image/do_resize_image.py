import os
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.transform import resize

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def cast_resized_image(image, dtype):
    if dtype == np.bool_:
        return image >= 0.5
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(image, info.min, info.max).astype(dtype)
    return image.astype(dtype)


def resize_image(image, scale_factor):
    original_dtype = image.dtype
    is_binary = original_dtype == np.bool_
    new_shape = (
        max(1, int(round(image.shape[0] * scale_factor))),
        max(1, int(round(image.shape[1] * scale_factor)))
    )

    resized = resize(
        image,
        new_shape,
        order=0 if is_binary else 1,
        anti_aliasing=not is_binary,
        preserve_range=True
    )
    return cast_resized_image(resized, original_dtype)


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = sys.argv[5]
scale_factor = float(sys.argv[6])

print("INPUT_DIR", INPUT_DIR)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)
print("scale_factor", scale_factor)

input_file = os.path.join(
    INPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

print("Running image resize")
img = tifffile.imread(input_file)
img = resize_image(img, scale_factor)
tifffile.imwrite(output_file, img)
print("Done")
