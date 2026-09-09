import json
import os
import sys
from pathlib import Path

import tifffile
import numpy as np

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.z_range import resolve_tiff_path

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
z = int(sys.argv[5])


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
    f"r{resolution_level:02d}_t00_c{channel:02d}_z{z:04d}.npy"
)

if not os.path.exists(output_file):
    img = tifffile.imread(input_file)
    np.save(output_file, np.array([img.min(), img.max()]))
