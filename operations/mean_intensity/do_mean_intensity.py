import json
import math
import os
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import parse_slurm_errors, submit_slurm_indices

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


UM_COLUMNS = ["z_raw", "y_raw", "x_raw"]
AXIS_COLUMNS = ["axis-0", "axis-1", "axis-2"]


def partial_csv_path(partials_folder, z):
    return os.path.join(partials_folder, f'part_mean_intensity_z_{str(int(z)).zfill(5)}.csv')


def z_voxel_coordinates(df, resolution_z):
    # z coords rounded to integer voxel indices at the detection resolution level
    if all(column in df.columns for column in AXIS_COLUMNS):
        return np.round(df["axis-0"].to_numpy(dtype=float))
    if all(column in df.columns for column in UM_COLUMNS):
        return np.round(df["z_raw"].to_numpy(dtype=float) / resolution_z)
    raise ValueError(
        'CSV must contain either z_raw/y_raw/x_raw (microns) or axis-0/axis-1/axis-2 (voxels) columns'
    )


print("Doing mean intensity orchestration")

cells_path = sys.argv[1]
raw_tiff_dir = sys.argv[2]
raw_provenance_path = sys.argv[3]
results_folder = sys.argv[4]
radius = float(sys.argv[5])
username = sys.argv[6]
priority = sys.argv[7]

jobs_folder = os.path.join(results_folder, "slurm_jobs")
partials_folder = os.path.join(results_folder, "partial_dfs")
csv_basename = os.path.basename(cells_path.replace('.csv', ''))
final_csv_path = os.path.join(results_folder, f"{csv_basename}_mean_intensity_r{radius:g}.csv")
out_column = f"mean_intensity_r{radius:g}"

try:
    os.makedirs(partials_folder)
except:
    pass

if os.path.exists(final_csv_path):
    print("Previous mean intensity results found. Delete them if you want to re-run")
    sys.exit(0)

metadata = json.load(open(raw_provenance_path, 'r'))
resolution = [float(x) for x in metadata['resolution']]  # um per voxel, z/y/x
shape = metadata['shape']
n_planes = int(shape[-3])

print("input file", cells_path)
df = pd.read_csv(cells_path)
print("Read df with", df.shape[0], "rows")

z_voxels = z_voxel_coordinates(df, resolution[0])

# a z layer is processed only when the full sphere fits inside the volume
half_z = int(math.ceil(radius / resolution[0]))
valid_layers = sorted({
    int(z) for z in np.unique(z_voxels)
    if z - half_z >= 0 and z + half_z <= n_planes - 1
})
if not valid_layers:
    raise ValueError(
        f"No points lie in a z range where the full sphere of radius {radius} um "
        f"fits inside {n_planes} planes"
    )
print("z layers to process:", len(valid_layers), "of", len(np.unique(z_voxels)))

path_to_task = os.path.join(jobs_folder, "mean_intensity_z_layers.sh")
worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_mean_intensity_z_layer.py")
with open(path_to_task, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('\n')
    f.write(f"#SBATCH -J {username}-mean-intensity-z")
    f.write('\n')
    f.write(f"#SBATCH -o {jobs_folder}/mean_intensity_%A_%a.out")
    f.write('\n')
    f.write('\n')
    f.write("source /h20/home/lab/miniconda3/bin/activate peace")
    f.write('\n')
    f.write(f'python {worker_script}')
    f.write(' ')
    f.write(raw_tiff_dir if ' ' not in raw_tiff_dir else f'"{raw_tiff_dir}"')
    f.write(' ')
    f.write(results_folder if ' ' not in results_folder else f'"{results_folder}"')
    f.write(' ')
    f.write(cells_path if ' ' not in cells_path else f'"{cells_path}"')
    f.write(' ')
    f.write(raw_provenance_path if ' ' not in raw_provenance_path else f'"{raw_provenance_path}"')
    f.write(' ')
    f.write(str(radius))
    f.write(' ')
    f.write('$SLURM_ARRAY_TASK_ID')
    f.write('\n')

existing = {z for z in valid_layers if os.path.exists(partial_csv_path(partials_folder, z))}
missing = len(valid_layers) - len(existing)
if missing:
    print(f"Submitting {missing} missing z-layer jobs of {len(valid_layers)}")
    job_ids = submit_slurm_indices(
        path_to_task,
        valid_layers,
        existing_outputs=existing,
        partition=settings.SLURM_PARTITION_CPU,
        cores=4,
        memory=32,
        priority=priority,
    )
    print("Submitted job ids:", job_ids)
else:
    print("All partial CSVs already exist")

while any(not os.path.exists(partial_csv_path(partials_folder, z)) for z in valid_layers):
    done = sum(1 for z in valid_layers if os.path.exists(partial_csv_path(partials_folder, z)))
    print(f"Partial CSVs done: {done} of {len(valid_layers)}")
    time.sleep(60)

# merge all partial z-layer CSVs into a single final CSV
partial_dfs = [pd.read_csv(partial_csv_path(partials_folder, z)) for z in valid_layers]
merged = pd.concat(partial_dfs, ignore_index=True)
print("Final merged df shape", merged.shape)
merged.to_csv(final_csv_path, index=False)
print("Final CSV saved as", final_csv_path)

has_errors = parse_slurm_errors(jobs_folder)
print("Errors found:", has_errors)
