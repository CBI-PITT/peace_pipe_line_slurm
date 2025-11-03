import os
import sys
from glob import glob

import tifffile
import numpy as np

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
z = int(sys.argv[5])

# input_file = os.path.join(
#     INPUT_DIR,
#     f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
# )

files = sorted(glob(os.path.join(INPUT_DIR, "*.tif")))
input_file = files[z]
print("Input file:", input_file)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.npy"
)

if not os.path.exists(output_file):
    img = tifffile.imread(input_file)
    np.save(output_file, np.array([img.min(), img.max()]))
