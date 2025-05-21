import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import tifffile
from ilastik.experimental.api import from_project_file
from xarray import DataArray

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
model_path = sys.argv[5]
z = int(sys.argv[6])
binarize_threshold = float(sys.argv[7])

print("INPUT_DIR", INPUT_DIR)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)


def binarize(image):
    pipeline = from_project_file(model_path)

    prediction = pipeline.predict(DataArray(image, dims=("y", "x")))
    image = np.array(prediction)[:, :, 0]
    return image


# input_file = os.path.join(
#     INPUT_DIR,
#     f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
# )
input_files = sorted(glob(os.path.join(INPUT_DIR, "*.tif")))
input_file = input_files[z]


output_file = os.path.join(
    OUTPUT_DIR,
    f'ilastik',
    f'resolution_level_{resolution_level}',
    f'channel_{channel}',
    f"ilastik_model_{os.path.basename(model_path).replace('.ilp', '')}",
    os.path.basename(input_file)
    # f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

if os.path.exists(output_file):
    sys.exit(0)

print("Running ilastik")
img = tifffile.imread(input_file)
try:
    img = binarize(img)
except:
    img = np.zeros(img.shape, img.dtype)
    import traceback
    print(traceback.format_exc())

tifffile.imwrite(output_file, img)

output_file_binary = os.path.join(
    OUTPUT_DIR,
    f'ilastik',
    f'resolution_level_{resolution_level}',
    f'channel_{channel}',
    f"ilastik_model_{os.path.basename(model_path).replace('.ilp', '')}_threshold_{binarize_threshold}",
    os.path.basename(input_file)
)

mask = img > binarize_threshold
tifffile.imwrite(output_file_binary, mask)

print("Done")
