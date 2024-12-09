import os
import sys

import numpy as np
import tifffile
from ilastik.experimental.api import from_project_file
from xarray import DataArray


IMS_FILE_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
model_path = sys.argv[5]
z = sys.argv[6]

print("IMS_FILE_PATH", IMS_FILE_PATH)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)


def binarize(image):
    pipeline = from_project_file(model_path)

    prediction = pipeline.predict(DataArray(image, dims=("y", "x")))
    image = np.array(prediction)[:, :, 0]
    return image


input_file = os.path.join(
    OUTPUT_DIR,
    f'resolution_level_{resolution_level}',
    f'channel_{channel}',
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_file = os.path.join(
    OUTPUT_DIR,
    f'ilastik',
    f"r{str(resolution_level)}_c{str(channel)}_ilastik_model_{os.path.basename(model_path)}",
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

print("Running ilastik")
img = tifffile.imread(input_file)
img = binarize(img)
tifffile.imwrite(output_file, img)
print("Done")
