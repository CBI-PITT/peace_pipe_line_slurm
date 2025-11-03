import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


print("Converting to napari")

number = sys.argv[1]
print("number", number)
detection_folder = sys.argv[2]
print("detection_folder", detection_folder)
napari_folder = sys.argv[3]
print("napari folder", napari_folder)


def save_empty_napari_df():
    print(f"WARNING: saving empty napari DF for chunk # {number}")
    napari_csv_file_path = os.path.join(napari_folder, f"napari_chunk_{str(number).zfill(5)}.csv")
    df = pd.DataFrame()
    df.to_csv(napari_csv_file_path)


csv_file = os.path.join(detection_folder, f"chunk_{str(number).zfill(5)}.csv")
napari_csv_file_path = os.path.join(napari_folder, f"napari_{os.path.basename(csv_file)}")
if os.path.exists(napari_csv_file_path):
    sys.exit(0)
try:
    df = pd.read_csv(csv_file)
except pd.errors.EmptyDataError as e:
    print("Warning: ", e)
    df2 = pd.DataFrame()
    df2.to_csv(napari_csv_file_path)
    sys.exit(0)

try:
    df2 = pd.DataFrame()
    df2['index'] = list(range(df.shape[0]))
    zvals = df['z'].tolist()
    yvals = df['y [px]'].tolist()
    xvals = df['x [px]'].tolist()
    df2['axis-0'] = zvals
    df2['axis-1'] = xvals
    df2['axis-2'] = yvals
    df2.to_csv(napari_csv_file_path)
except:
    print(f"EXCEPTION: unable to save to napari format chunk # {number}")
    print(traceback.format_exc())
    save_empty_napari_df()
