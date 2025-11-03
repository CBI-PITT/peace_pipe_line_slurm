"""
$ conda activate spotiflow
 - take 2D .tif image (path), output folder, model name, and z (for 3D images processed in layers)
 - return 3D coordinates as napari-style csv (columns: axis-0, axis-1, axis-2). all axis-0 values = z
"""

import os
import sys
from glob import glob

import pandas as pd
from spotiflow.model import Spotiflow
import tifffile
from pathlib import Path
this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


input_dir = sys.argv[1]
output_dir = sys.argv[2]
model_name = sys.argv[3]
z = int(sys.argv[4])

files = sorted(glob(os.path.join(input_dir, "*.tif")))
img_path = files[z]

model = Spotiflow.from_pretrained(model_name)

img = tifffile.imread(img_path)

points, details = model.predict(img)
z_values = [z] * len(points)

df = pd.DataFrame()
df['axis-0'] = z_values
df['axis-1'] = points[:,0]
df['axis-2'] = points[:,1]
df.to_csv(os.path.join(output_dir, os.path.basename(img_path.replace('.tif', '.csv'))))
