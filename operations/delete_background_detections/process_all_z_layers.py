import json
import os
import re
import subprocess
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
from utils.slurm import split_slurm_array, submit_slurm_array

os.umask(settings.UMASK)


def launch_job_array():
    path_to_task = os.path.join(
        jobs_folder,
        f"delete_background_detections_rl{resolution_level}_c{channel}.sh"
    )
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "process_z_layer.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-delete-bg-detections")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_delete_bg_detections_%A_%a.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(input_tiff_stack_dir)
        f.write(' ')
        f.write(save_folder)
        f.write(' ')
        f.write(points_df)
        f.write(' ')
        f.write(masks_dir)
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write('\n')

    already_done = glob(os.path.join(save_folder, "*.csv"))
    if len(already_done):
        print("Partially processed")
        print("Processed", len(already_done), "of", z_layers)
        files = os.listdir(save_folder)
        pattern = "_z(\d+)\.csv"
        numbers = [re.findall(pattern, x)[0] for x in files]
        numbers = set(map(int, numbers))
        split_slurm_array(
            path_to_task,
            z_layers,
            numbers,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=12,
            memory=32,
            priority=priority
        )
    else:
        submit_slurm_array(
            path_to_task,
            z_layers,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=12,
            memory=32,
            priority=priority
    )


def merge_df():
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(save_folder, '*.csv')))
    # print("CSV files", len(csv_files), csv_files[:3])
    for z in csv_files:
        print("Processing", z)
        chunk_df = pd.read_csv(z)
        if chunk_df.empty:
            continue
        df = pd.concat([df, chunk_df])

    print("Saving df")
    df.to_csv(os.path.join(str(Path(save_folder).parent), f'{os.path.basename(save_folder)}.csv'), index=False)


input_tiff_stack_dir = sys.argv[1]
save_folder = sys.argv[2]
points_df = sys.argv[3]
masks_dir = sys.argv[4]
username = sys.argv[6]
priority = sys.argv[7]

metadata = json.load(open(os.path.join(input_tiff_stack_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
z_layers = metadata['shape'][-3]
resolution_level = metadata['resolution_level']
channel = metadata['channel']

output_folder_sequence = Path(save_folder).parent
jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")

launch_job_array()

layers_done = len(glob(os.path.join(save_folder, "*.csv")))
while layers_done < z_layers:
    print(f"Layers done: {layers_done} of {z_layers}")
    time.sleep(120)
    layers_done = len(glob(os.path.join(save_folder, "*.csv")))

# merge the dataframes with points for all chunks
merge_df()
