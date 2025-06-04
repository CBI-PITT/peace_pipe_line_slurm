import json
import os
import re
import subprocess
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import tifffile

from pathlib import Path
this_script = Path(__file__)
operation_folder = this_script.parent
operations_folder = operation_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import split_slurm_array, submit_slurm_array


def find_min_max(pth):
    imgs = sorted(glob(os.path.join(pth, "*.tif")))
    img_stack = np.stack([tifffile.imread(img) for img in imgs])
    return img_stack.min(), img_stack.max()


def find_min_max_imaris(pth):
    from imaris_ims_file_reader import ims
    f = ims(pth)
    return f.metaData[(resolution_level, 0, channel, 'HistogramMin')], f.metaData[(resolution_level, 0, channel, 'HistogramMax')]

def submit_denoising_job_array():
    z_layers = metadata['shape'][-3]
    path_to_task = os.path.join(jobs_folder, f"denoise_cellpose_rl{resolution_level}_c{channel}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "do_denoising.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-denoise-cellpose")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate cellpose")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(INPUT_DIR if ' ' not in INPUT_DIR else f'"{INPUT_DIR}"')
        f.write(' ')
        f.write(save_folder if ' ' not in save_folder else f'"{save_folder}"')
        f.write(' ')
        f.write(str(resolution_level))
        f.write(' ')
        f.write(str(channel))
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write(' ')
        f.write(str(model))
        f.write(' ')
        f.write(str(diameter))
        f.write(' ')
        f.write(str(stack_min))
        f.write(' ')
        f.write(str(stack_max))
        f.write('\n')

    already_done = glob(os.path.join(save_folder, "*.tif"))
    if len(already_done):
        print("Partially processed")
        print("Processed", len(already_done), "of", z_layers)
        files = os.listdir(save_folder)
        pattern = "_z(\d+)\.tif"
        numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
        numbers = set(map(int, numbers))
        job_ids = split_slurm_array(
            path_to_task,
            z_layers,
            numbers,
            partition=settings.SLURM_PARTITION_GPU,
            cores=1,
            memory=32,
            needs_gpu=True,
            priority=priority,
        )
    else:
        job_ids = submit_slurm_array(
            path_to_task,
            z_layers,
            partition=settings.SLURM_PARTITION_GPU,
            cores=1,
            memory=32,
            needs_gpu=True,
            priority=priority,
        )
    return job_ids


def submit_conversion_job_array():
    z_layers = metadata['shape'][-3]
    path_to_task = os.path.join(jobs_folder, f"denoise_cellpose_rl{resolution_level}_c{channel}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "save_as_uint.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-denoise-cellpose-convert")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(save_folder if ' ' not in save_folder else f'"{save_folder}"')
        f.write(' ')
        f.write(save_folder_uint if ' ' not in save_folder_uint else f'"{save_folder_uint}"')
        f.write(' ')
        f.write(str(resolution_level))
        f.write(' ')
        f.write(str(channel))
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write(' ')
        f.write(str(denoised_stack_min))
        f.write(' ')
        f.write(str(denoised_stack_max))
        f.write('\n')

    already_done = glob(os.path.join(save_folder_uint, "*.tif"))
    if len(already_done):
        print("Partially processed")
        print("Processed", len(already_done), "of", z_layers)
        files = os.listdir(save_folder)
        pattern = "_z(\d+)\.tif"
        numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
        numbers = set(map(int, numbers))
        job_ids = split_slurm_array(
            path_to_task,
            z_layers,
            numbers,
            partition=settings.SLURM_PARTITION_CPU,
            cores=1,
            memory=32,
            priority=priority,
        )
    else:
        job_ids = submit_slurm_array(
            path_to_task,
            z_layers,
            partition=settings.SLURM_PARTITION_CPU,
            cores=1,
            memory=32,
            priority=priority,
        )
    return job_ids


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
username = sys.argv[5]
priority = sys.argv[6]
model = sys.argv[7]
diameter = int(sys.argv[8])

metadata = json.load(open(os.path.join(INPUT_DIR, f'.{settings.INFO_FILE_NAME}'), 'r'))

output_operation_folder = os.path.join(OUTPUT_DIR, "denoise_cellpose")
jobs_folder = os.path.join(output_operation_folder, "slurm_jobs")
save_folder = os.path.join(
    output_operation_folder,
    f'resolution_level_{resolution_level}',
    f'channel_{channel}',
    f"cellpose_model_{model}_diameter_{diameter}"
)

base_input_dir = metadata['base_input_dir']
if not base_input_dir:
    base_input_dir = INPUT_DIR
base_metadata = json.load(open(os.path.join(base_input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
source_path = base_metadata['source']
if source_path.endswith('.ims'):
    stack_min, stack_max = find_min_max_imaris(source_path)
    print("Imaris stack_min", stack_min)
    print("Imaris stack_max", stack_max)

stack_min, stack_max = find_min_max(INPUT_DIR)  # TODO: store to a text file
print("Calculated stack_min", stack_min)
print("Calculated stack_max", stack_max)

submit_denoising_job_array()

finished_planes = len(glob(os.path.join(save_folder, '*.tif')))
z_layers = metadata['shape'][-3]
while finished_planes < z_layers:
    print("finished", finished_planes, "of", z_layers)
    time.sleep(60)
    finished_planes = len(glob(os.path.join(save_folder, '*.tif')))

denoised_stack_min, denoised_stack_max = find_min_max(save_folder)  # TODO: store to a text file
save_folder_uint = save_folder + "_uint"
if not os.path.exists(save_folder_uint):
    os.makedirs(save_folder_uint)

submit_conversion_job_array()

finished_planes = len(glob(os.path.join(save_folder_uint, '*.tif')))
z_layers = metadata['shape'][-3]
while finished_planes < z_layers:
    print("finished", finished_planes, "of", z_layers)
    time.sleep(10)
    finished_planes = len(glob(os.path.join(save_folder_uint, '*.tif')))
