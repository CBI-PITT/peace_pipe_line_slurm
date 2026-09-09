import os
import sys

import tifffile
import numpy as np

from pathlib import Path
this_script = Path(__file__)
operation_folder = this_script.parent
operations_folder = operation_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
z = sys.argv[5]
denoised_stack_min = float(sys.argv[6])
denoised_stack_max = float(sys.argv[7])

input_file = os.path.join(
    INPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

output_file = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)

img_dn = tifffile.imread(input_file)
if denoised_stack_max == denoised_stack_min:
    img_dn = np.zeros_like(img_dn, dtype=np.float32)
else:
    img_dn = (img_dn - denoised_stack_min) / (denoised_stack_max - denoised_stack_min)
img_dn = img_dn * 65535
tifffile.imwrite(output_file, img_dn.astype('uint16'))
