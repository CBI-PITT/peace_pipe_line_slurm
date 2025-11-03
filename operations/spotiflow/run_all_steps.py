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
from dask import delayed, compute
import dask

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import split_slurm_array, submit_slurm_array, parse_slurm_errors, submit_slurm_job

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def launch_job_array():
    path_to_task = os.path.join(
        jobs_folder,
        f"spotiflow_rl{resolution_level}_c{channel}.sh"
    )
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "actual_operation.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-spotiflow-z")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/spotiflow_%A_%a.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate spotiflow")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(input_tiff_stack_dir)
        f.write(' ')
        f.write(save_folder)
        f.write(' ')
        f.write(model_name)
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write('\n')

    already_done = glob(os.path.join(save_folder, "*.csv"))
    if len(already_done):
        print("Partially processed")
        print("Processed", len(already_done), "of", z_layers)
        files = os.listdir(save_folder)
        pattern = "_z(\d+)\.csv"
        numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.csv')]
        numbers = set(map(int, numbers))
        split_slurm_array(
            path_to_task,
            z_layers,
            numbers,
            partition=settings.SLURM_PARTITION_EXTREME,
            needs_gpu=True,
            cores=2,
            memory=32,
            priority=priority
        )
    else:
        submit_slurm_array(
            path_to_task,
            z_layers,
            partition=settings.SLURM_PARTITION_EXTREME,
            needs_gpu=True,
            cores=2,
            memory=32,
            priority=priority
    )


def merge_df():
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(save_folder, '*.csv')))
    delayed_tasks = [delayed(pd.read_csv)(z) for z in csv_files]
    results = compute(*delayed_tasks)
    df = pd.concat(results, ignore_index=True)
    print("Saving df")
    df.to_csv(os.path.join(str(Path(save_folder).parent), f'{model_name}_merged_df.csv'), index=False)


def start_dbscan_job():
    path_to_task = os.path.join(jobs_folder, f"dbscan.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "run_dbscan.py")
    input_file = os.path.join(str(Path(save_folder).parent), f'{model_name}_merged_df.csv')
    output_dir = str(output_folder_sequence)
    try:
        os.makedirs(output_dir)
    except:
        pass

    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-spotiflow-dbscan")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/logs_dbscan/slurm_spotiflow_dbscan.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate dbscan")
        f.write('\n')
        f.write(f'python {slurm_script} ')
        f.write(input_file if ' ' not in input_file else f'"{input_file}"')
        f.write(' ')
        f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
        f.write(' ')
        f.write(model_name)
        f.write('\n')

    submit_slurm_job(
        path_to_task,
        partition=settings.SLURM_PARTITION_HIGH_RAM,
        cores=24,
        memory=128,
        priority=priority,
    )


input_tiff_stack_dir = sys.argv[1]
save_folder = sys.argv[2]
username = sys.argv[3]
priority = sys.argv[4]
model_name = sys.argv[5]
with_dbscan = int(sys.argv[6])

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

if with_dbscan:
    start_dbscan_job()

    wait_iterations = 1000  # 1000 x 1 minute = ~16h
    current_iteration = 0
    while current_iteration < wait_iterations:
        dbscan_df = os.path.join(output_folder_sequence, f"{model_name}_merged_dbscan_df.csv")
        if os.path.exists(dbscan_df):
            break
        time.sleep(60)
        current_iteration += 1

has_errors = parse_slurm_errors(jobs_folder)
print("Errors found:", has_errors)
