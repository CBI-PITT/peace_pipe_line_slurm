import os
import sys
from pathlib import Path

import tifffile
from cellpose import models

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


input_file = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
model = sys.argv[5]
diameter = int(sys.argv[6])

print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("model", model)
print("diameter", diameter)


def segment(image, model, diameter):
    model = models.CellposeModel(gpu=True, model_type=model)
    mask, _, _ = model.eval(image, do_3D=False, diameter=diameter, stitch_threshold=0.4, anisotropy=4.5)
    return mask


output_file = os.path.join(
    OUTPUT_DIR,
    # f'cellpose',
    # f'resolution_level_{resolution_level}',
    # f'channel_{channel}',
    # f"cellpose_model_{model}",
    f"mask_{os.path.basename(input_file)}"
)

if os.path.exists(output_file):
    print(f"file {output_file} already exists")
    sys.exit(0)

print("Running cellpose")
img = tifffile.imread(input_file)
img = segment(img, model, diameter)
tifffile.imwrite(output_file, img)
print("Done")
