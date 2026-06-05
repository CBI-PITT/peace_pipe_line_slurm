from glob import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

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
output_dir = sys.argv[2]  # output/resnet_classification/resolution_level_<>/channel_<>/
cell_candidates_path = sys.argv[3]  # .csv
model_path = sys.argv[4]  # pytorch
username = sys.argv[5]
priority = sys.argv[6]

metadata_path = os.path.join(input_dir, '.dataset_info.json')
metadata = json.load(open(metadata_path, 'r'))
model_name = os.path.basename(model_path)

jobs_folder = os.path.join(output_dir, "slurm_jobs")
try:
    os.makedirs(jobs_folder)
except:
    pass

save_results_to = os.path.join(output_dir, 'points')
try:
    os.makedirs(save_results_to)
except:
    pass

if os.path.exists(
    os.path.join(
        save_results_to,
        f'predicted_non_cells_{model_name}.csv'
    )
) and os.path.exists(
    os.path.join(
        save_results_to,
        f'predicted_cells_{model_name}.csv'
    )
):
    print("Previous classification results found. Delete them if you want to re-run the classification")
    sys.exit(0)

points_df = pd.read_csv(cell_candidates_path)
print("napari csv shape", points_df.shape)
n_cubes = points_df.shape[0]

df_inference = points_df.copy()
df_inference.rename(columns={'axis-0': 'axis_0', 'axis-1': 'axis_1', 'axis-2': 'axis_2'}, inplace=True)
df_inference['ann'] = ['unknown1'] * (n_cubes // 2) + ['unknown2'] * (n_cubes - n_cubes // 2)

# Extract cubes the same way it is done in cellfinder
cube_shape2 = (20, 25, 25)  # TODO: this should be in physical units, not pixels
img_shape = metadata['shape']

df_low_z = points_df[points_df['axis-0'] < cube_shape2[0] // 2].copy()
print("df_low_z", df_low_z.shape)
df_low_z['nn_decoded'] = [''] * df_low_z.shape[0]
df_low_z['prob'] = [np.nan] * df_low_z.shape[0]
df_low_z['incomplete'] = [True] * df_low_z.shape[0]

print("Last z classified", img_shape[0] - cube_shape2[0] // 2)
df_high_z = points_df[points_df['axis-0'] > img_shape[0] - cube_shape2[0] // 2].copy()
print("df_high_z", df_high_z.shape)
df_high_z['nn_decoded'] = [''] * df_high_z.shape[0]
df_high_z['prob'] = [np.nan] * df_high_z.shape[0]
df_high_z['incomplete'] = [True] * df_high_z.shape[0]

filenames = sorted(glob(os.path.join(input_dir, '*.tif')))
print("filenames", len(filenames))

z_layers = metadata['shape'][-3]
path_to_task = os.path.join(jobs_folder, f"resnet_z_layers_gpu.sh")
main_script = os.path.abspath(__file__)
slurm_script = os.path.join(os.path.dirname(main_script), "run_classification_z_layer.py")
from utils.containers import build_container_exec_prefix

with open(path_to_task, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('\n')
    f.write(f"#SBATCH -J {username}-resnet-gpu")
    f.write('\n')
    f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
    f.write('\n')
    f.write('\n')
    f.write(build_container_exec_prefix('resnet_classification', os.path.dirname(main_script), needs_gpu=True))
    f.write(f' python {slurm_script}')
    f.write(' ')
    f.write(input_dir if ' ' not in input_dir else f'"{input_dir}"')
    f.write(' ')
    f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
    f.write(' ')
    f.write(','.join(list(map(str, cube_shape2))))
    f.write(' ')
    f.write(cell_candidates_path if ' ' not in cell_candidates_path else f'"{cell_candidates_path}"')
    f.write(' ')
    f.write(model_path if ' ' not in model_path else f'"{model_path}"')
    f.write(' ')
    f.write('$SLURM_ARRAY_TASK_ID')
    f.write('\n')

df_inference_complete = []
total_rows = 0
z_center_min = cube_shape2[0] // 2
z_center_max = z_layers - cube_shape2[0] + cube_shape2[0] // 2

## run on GPU partition
command = [
    'sbatch',
    f'--array={z_center_min}-{z_center_max}',  # TODO use submit_slurm_array
    '-p', 'gpu',
    '--gres=gpu:1',
    '--mem=64Gb',
    '-n8',
    f'--nice={settings.PRIORITY_TO_NICE_MAP_GPU[priority]}',
    path_to_task
]
subprocess.run(command)

## wait for all jobs to finish
expected_dfs = len(list(range(z_center_min, z_center_max+1)))
print("Expected DFs", expected_dfs)
finished = len(glob(os.path.join(output_dir, 'partial_dfs', 'part_classification_df_z_*.csv')))
while finished < expected_dfs:
    time.sleep(30)
    finished = len(glob(os.path.join(output_dir, 'partial_dfs', 'part_classification_df_z_*.csv')))
    print("finished DFs", finished)

for partial_df_path in sorted(glob(os.path.join(output_dir, 'partial_dfs', 'part_classification_df_z_*.csv'))):
    partial_df = pd.read_csv(partial_df_path)
    df_inference_complete.append(partial_df)
    total_rows += partial_df.shape[0]
    print("total_rows", total_rows)

df_inference_complete = pd.concat(df_inference_complete, ignore_index=True)

print("Shape of df without leading and trailing z layers", df_inference_complete.shape)

df_inference_complete = pd.concat([df_low_z, df_inference_complete, df_high_z], ignore_index=True)
print("Final predictions df", df_inference_complete.shape)
print("Saving outputs")
df_inference_complete.to_csv(
    os.path.join(
        save_results_to,
        f'predictions_{model_name}.csv'
    ),
    index=False
)

df_inference_cells = df_inference_complete.loc[df_inference_complete["nn_decoded"] == "cell"]
df_inference_non_cells = df_inference_complete.loc[df_inference_complete["nn_decoded"] == "non_cell"]

# save classification results to napari compatible csv
cells_df = pd.DataFrame()
cells_df['index'] = list(range(df_inference_cells.shape[0]))
cells_df['axis-0'] = df_inference_cells['axis-0'].to_list()
cells_df['axis-1'] = df_inference_cells['axis-1'].to_list()
cells_df['axis-2'] = df_inference_cells['axis-2'].to_list()
cells_df.to_csv(
    os.path.join(
        save_results_to,
        f'predicted_cells_{model_name}.csv'
    ),
    index=False
)

non_cells_df = pd.DataFrame()
non_cells_df['index'] = list(range(df_inference_non_cells.shape[0]))
non_cells_df['axis-0'] = df_inference_non_cells['axis-0'].to_list()
non_cells_df['axis-1'] = df_inference_non_cells['axis-1'].to_list()
non_cells_df['axis-2'] = df_inference_non_cells['axis-2'].to_list()
non_cells_df.to_csv(
    os.path.join(
        save_results_to,
        f'predicted_non_cells_{model_name}.csv'
    ),
    index=False
)
