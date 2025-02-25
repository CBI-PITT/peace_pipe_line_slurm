import os
import sys

import tifffile
from cellpose import models


input_file = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
model = sys.argv[5]

print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("model", model)


def segment(image, model):
    model = models.CellposeModel(gpu=True, model_type=model)
    mask, _, _ = model.eval(image, do_3D=False, diameter=15, stitch_threshold=0.4, anisotropy=4.5)
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
img = segment(img, model)
tifffile.imwrite(output_file, img)
print("Done")
