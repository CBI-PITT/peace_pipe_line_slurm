import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd


INFO_FILE_NAME = "dataset_info.json"
PRIORITY_TO_NICE_MAP_COMPUTE = {
    '0': 1000000,  # priority 1 for compute n24 mem64;      priority 1 for GPU
    '1': 61500,    # priority 77 for compute n24 mem64;
    '2': 61000,    # priority 577 for compute n24 mem64;
    '3': 60000,    # priority 1577 for compute n24 mem64;
    '4': 50000,    # priority 11577 for compute n24 mem64;
    '5': 40000,     # priority 21577 for compute n24 mem64;
    '1313': 0
}


def launch_job_array():
    path_to_task = os.path.join(
        jobs_folder,
        f"delete_background_detections_rl{resolution_level}_c{channel}.sh"
    )
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "process_z_layer.py")
    nice_value = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-delete-bg-detections")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/delete_bg_detections_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(INPUT_TIFF_STACK_DIR)
        f.write(' ')
        f.write(OUTPUT_DIR)
        f.write(' ')
        f.write(POINTS_DF)
        f.write(' ')
        f.write(MASKS_DIR)
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write('\n')

    command = [
        'sbatch',
        f'--array=0-{z_layers - 1}',
        '-p', f'compute,gpu',
        '--mem=32Gb',
        '-n12',
        f'--nice={nice_value}',
        path_to_task
    ]
    subprocess.run(command)


def merge_df():
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(OUTPUT_DIR, '*.csv')))
    # print("CSV files", len(csv_files), csv_files[:3])
    for z in csv_files:
        print("Processing", z)
        chunk_df = pd.read_csv(z)
        if chunk_df.empty:
            continue
        df = pd.concat([df, chunk_df])

    print("Saving df")
    df.to_csv(os.path.join(str(Path(OUTPUT_DIR).parent), 'cleaned_bg_merged_df.csv'))


INPUT_TIFF_STACK_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
POINTS_DF = sys.argv[3]
MASKS_DIR = sys.argv[4]
GLOBAL_OUTPUT = sys.argv[5]
username = sys.argv[6]
priority = sys.argv[7]

metadata = json.load(open(os.path.join(INPUT_TIFF_STACK_DIR, f'.{INFO_FILE_NAME}'), 'r'))
z_layers = metadata['shape'][-3]
resolution_level = metadata['resolution_level']
channel = metadata['channel']

output_operation_folder = os.path.join(GLOBAL_OUTPUT, 'delete_background_detections')
jobs_folder = os.path.join(output_operation_folder, "slurm_jobs")

launch_job_array()

layers_done = len(glob(os.path.join(OUTPUT_DIR, "*.csv")))
while layers_done < z_layers:
    print(f"Layers done: {layers_done} of {z_layers}")
    time.sleep(120)
    layers_done = len(glob(os.path.join(OUTPUT_DIR, "*.csv")))

# merge the dataframes with points for all chunks
merge_df()
