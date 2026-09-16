import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


UM_COLUMNS = ["z_raw", "y_raw", "x_raw"]
AXIS_COLUMNS = ["axis-0", "axis-1", "axis-2"]
ATLAS_COLUMNS = ["atlas_structure_name", "atlas_structure_acronym", "atlas_structure_number"]
NN_COLUMN = "nearest_neighbor_distance_um"


def resolve_resolution(options):
    # provenance writers propagate resolution, but walk the source chain
    # when it is absent (cycle-safe)
    if options.get('resolution'):
        return options['resolution']
    source_path = options.get('source')
    seen = set()
    while source_path and source_path not in seen and os.path.exists(source_path):
        seen.add(source_path)
        with open(source_path, 'r') as f:
            source_data = json.load(f)
        if source_data.get('resolution'):
            return source_data['resolution']
        source_path = source_data.get('source')
    return None


def coordinates_in_um(df, resolution):
    if all(column in df.columns for column in UM_COLUMNS):
        return df[UM_COLUMNS].to_numpy(dtype=float), UM_COLUMNS
    if all(column in df.columns for column in AXIS_COLUMNS):
        if resolution is None:
            raise ValueError(
                'CSV uses axis-0/axis-1/axis-2 voxel coordinates but no resolution '
                'could be resolved from the provenance chain; cannot convert to microns'
            )
        return df[AXIS_COLUMNS].to_numpy(dtype=float) * np.asarray(resolution, dtype=float), AXIS_COLUMNS
    raise ValueError(
        'CSV must contain either z_raw/y_raw/x_raw (microns) or axis-0/axis-1/axis-2 (voxels) columns'
    )


def nearest_neighbor_distances(points_in_um):
    # distance to the closest other point for each row; NaN rows are excluded
    # from the tree and stay NaN
    distances = np.full(points_in_um.shape[0], np.nan)
    finite = ~np.any(~np.isfinite(points_in_um), axis=1)
    if np.count_nonzero(finite) < 2:
        return distances
    finite_indices = np.flatnonzero(finite)
    tree = cKDTree(points_in_um[finite])
    # k=2: the closest match is the point itself, the second is the true neighbor
    neighbor_distance, _ = tree.query(points_in_um[finite], k=2)
    distances[finite_indices] = neighbor_distance[:, 1]
    return distances


print("Doing nearest-neighbor distance calculation")

cells_path = sys.argv[1]
results_folder = sys.argv[2]
source_provenance_path = sys.argv[3]

print("input file", cells_path)
print("reading df")
df = pd.read_csv(cells_path)
print("Read df with", df.shape[0], "rows")

provenance = {}
if os.path.exists(source_provenance_path):
    with open(source_provenance_path, 'r') as f:
        provenance = json.load(f)
resolution = resolve_resolution(provenance)

points, coordinate_columns = coordinates_in_um(df, resolution)
print("Coordinates taken from columns", coordinate_columns)

distances = nearest_neighbor_distances(points)

out_df = df.copy()
out_df[NN_COLUMN] = distances

output_nn_csv_path = os.path.join(
    results_folder,
    f"{os.path.basename(cells_path.replace('.csv', ''))}_nn_distances.csv"
)
out_df.to_csv(output_nn_csv_path, index=False)
print('Nearest-neighbor CSV saved as', output_nn_csv_path)

# Per-region summary: only when the CSV carries atlas-region columns.
# Outside-atlas points (empty acronym) are excluded; mean/median skip rows
# with no NN distance (e.g. the single point of the whole dataset).
if all(column in df.columns for column in ATLAS_COLUMNS):
    region_df = out_df.copy()
    region_df["atlas_structure_acronym"] = region_df["atlas_structure_acronym"].fillna("")
    region_df = region_df[region_df["atlas_structure_acronym"] != ""]
    region_stats_df = (
        region_df
        .groupby(
            ["atlas_structure_name", "atlas_structure_acronym", "atlas_structure_number"],
            as_index=False,
        )
        .agg(
            cell_count=(NN_COLUMN, "size"),
            mean_nn_distance_um=(NN_COLUMN, "mean"),
            median_nn_distance_um=(NN_COLUMN, "median"),
        )
        .sort_values("cell_count", ascending=False)
    )
    output_region_csv_path = os.path.join(
        results_folder,
        f"{os.path.basename(cells_path.replace('.csv', ''))}_nn_region_stats.csv"
    )
    region_stats_df.to_csv(output_region_csv_path, index=False)
    print('Region nearest-neighbor stats CSV saved as', output_region_csv_path)
