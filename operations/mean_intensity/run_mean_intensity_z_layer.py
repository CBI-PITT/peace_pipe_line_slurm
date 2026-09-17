import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.z_range import resolve_tiff_path

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


UM_COLUMNS = ["z_raw", "y_raw", "x_raw"]
AXIS_COLUMNS = ["axis-0", "axis-1", "axis-2"]


def voxel_coordinates(df, resolution):
    # points in voxel indices at the detection resolution level
    if all(column in df.columns for column in AXIS_COLUMNS):
        return (
            np.round(df["axis-0"].to_numpy(dtype=float)),
            df["axis-1"].to_numpy(dtype=float),
            df["axis-2"].to_numpy(dtype=float),
        )
    if all(column in df.columns for column in UM_COLUMNS):
        return (
            np.round(df["z_raw"].to_numpy(dtype=float) / resolution[0]),
            df["y_raw"].to_numpy(dtype=float) / resolution[1],
            df["x_raw"].to_numpy(dtype=float) / resolution[2],
        )
    raise ValueError(
        'CSV must contain either z_raw/y_raw/x_raw (microns) or axis-0/axis-1/axis-2 (voxels) columns'
    )


def read_tiff_plane(raw_tiff_dir, metadata, resolution_level, channel, z):
    return tifffile.imread(resolve_tiff_path(raw_tiff_dir, metadata, resolution_level, channel, z))


raw_tiff_dir = sys.argv[1]
results_folder = sys.argv[2]
cells_path = sys.argv[3]
raw_provenance_path = sys.argv[4]
radius = float(sys.argv[5])
z_center = int(sys.argv[6])

partials_folder = os.path.join(results_folder, "partial_dfs")
try:
    os.makedirs(partials_folder)
except:
    pass

partial_path = os.path.join(partials_folder, f'part_mean_intensity_z_{str(z_center).zfill(5)}.csv')
if os.path.exists(partial_path):
    print("Partial CSV already exists for z", z_center)
    sys.exit(0)

metadata = json.load(open(raw_provenance_path, 'r'))
resolution = [float(x) for x in metadata['resolution']]  # um per voxel, z/y/x
shape = metadata['shape']
n_planes = int(shape[-3])
n_y = int(shape[-2])
n_x = int(shape[-1])
resolution_level = int(metadata['resolution_level'])
channel = int(metadata['channel'])
out_column = f"mean_intensity_r{radius:g}"

print("input file", cells_path)
df = pd.read_csv(cells_path)
print("Read df with", df.shape[0], "rows")

z_voxels, y_voxels, x_voxels = voxel_coordinates(df, resolution)
mask = z_voxels.astype(int) == z_center
layer_df = df[mask].reset_index(drop=True)
ys = np.round(y_voxels[mask]).astype(int)
xs = np.round(x_voxels[mask]).astype(int)
print("Points in z layer", z_center, ":", layer_df.shape[0])

# z planes per sphere radius on each side; the same in y/x for the kernel
half_z = int(math.ceil(radius / resolution[0]))
half_y = int(math.ceil(radius / resolution[1]))
half_x = int(math.ceil(radius / resolution[2]))

# sphere kernel: voxel offsets whose physical distance from the center is
# within the requested radius
offsets = []
for dz in range(-half_z, half_z + 1):
    for dy in range(-half_y, half_y + 1):
        for dx in range(-half_x, half_x + 1):
            dist_sq = (
                (dz * resolution[0]) ** 2
                + (dy * resolution[1]) ** 2
                + (dx * resolution[2]) ** 2
            )
            if dist_sq <= radius * radius:
                offsets.append((dz, dy, dx))
print("Sphere kernel size:", len(offsets), "voxels")

# read only the z planes needed for the sphere
tiff_stack = np.stack([
    read_tiff_plane(raw_tiff_dir, metadata, resolution_level, channel, z)
    for z in range(z_center - half_z, z_center + half_z + 1)
])
print("Read tiff stack", tiff_stack.shape)

# mean intensity per point; spheres crossing the y/x borders use the
# in-bounds voxels only (partial sphere)
n_points = layer_df.shape[0]
sums = np.zeros(n_points, dtype=float)
counts = np.zeros(n_points, dtype=int)
for dz, dy, dx in offsets:
    plane = tiff_stack[dz + half_z]
    y_idx = ys + dy
    x_idx = xs + dx
    valid = (y_idx >= 0) & (y_idx < n_y) & (x_idx >= 0) & (x_idx < n_x)
    if not valid.any():
        continue
    sums[valid] += plane[y_idx[valid], x_idx[valid]]
    counts[valid] += 1

means = np.full(n_points, np.nan)
has_values = counts > 0
means[has_values] = sums[has_values] / counts[has_values]

layer_df[out_column] = means
layer_df.to_csv(partial_path, index=False)
print("Partial CSV saved as", partial_path)
