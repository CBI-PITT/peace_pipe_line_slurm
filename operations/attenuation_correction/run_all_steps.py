import json
import os
import re
import shutil
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
from utils.slurm import split_slurm_array, submit_slurm_array, parse_slurm_errors, submit_slurm_job

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def submit_correction_job_array():
    print("Submitting job array")
    path_to_task = os.path.join(jobs_folder, f"attenuation_corr_rl{resolution_level}_c{channel}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "do_correction.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-attenuation-corr-z")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/attenuation_corr_z_slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(input_dir if ' ' not in input_dir else f'"{input_dir}"')
        f.write(' ')
        f.write(save_folder if ' ' not in save_folder else f'"{save_folder}"')
        f.write(' ')
        f.write(str(resolution_level))
        f.write(' ')
        f.write(str(channel))
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write(' ')
        f.write(str(reference_z))
        f.write(' ')
        f.write(str(opening_radius))
        f.write(' ')
        f.write(str(ref_mean))
        f.write(' ')
        f.write(str(ref_std))
        # f.write(' ')
        # f.write(str(min_max_folder_denoised))
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
            partition=settings.SLURM_PARTITION_HIGH_RAM,
            cores=1,
            memory=32,
            needs_gpu=False,
            priority=priority,
        )
    else:
        job_ids = submit_slurm_array(
            path_to_task,
            z_layers,
            partition=settings.SLURM_PARTITION_HIGH_RAM,
            cores=1,
            memory=32,
            needs_gpu=False,
            priority=priority,
        )
    return job_ids


# def submit_conversion_job_array():
#     # z_layers = metadata['shape'][-3]
#     path_to_task = os.path.join(jobs_folder, f"cellpose_convert_rl{resolution_level}_c{channel}.sh")
#     main_script = os.path.abspath(__file__)
#     slurm_script = os.path.join(os.path.dirname(main_script), "save_as_uint.py")
#     with open(path_to_task, 'w') as f:
#         f.write('#!/bin/bash\n')
#         f.write('\n')
#         f.write(f"#SBATCH -J {username}-denoise-cellpose-convert")
#         f.write('\n')
#         f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
#         f.write('\n')
#         f.write('\n')
#         f.write("source /h20/home/lab/miniconda3/bin/activate peace")
#         f.write('\n')
#         f.write(f'python {slurm_script}')
#         f.write(' ')
#         f.write(save_folder if ' ' not in save_folder else f'"{save_folder}"')
#         f.write(' ')
#         f.write(save_folder_uint if ' ' not in save_folder_uint else f'"{save_folder_uint}"')
#         f.write(' ')
#         f.write(str(resolution_level))
#         f.write(' ')
#         f.write(str(channel))
#         f.write(' ')
#         f.write('$SLURM_ARRAY_TASK_ID')
#         f.write(' ')
#         f.write(str(denoised_stack_min))
#         f.write(' ')
#         f.write(str(denoised_stack_max))
#         f.write('\n')
#
#     already_done = glob(os.path.join(save_folder_uint, "*.tif"))
#     if len(already_done):
#         print("Partially processed")
#         print("Processed", len(already_done), "of", z_layers)
#         files = os.listdir(save_folder)
#         pattern = "_z(\d+)\.tif"
#         numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
#         numbers = set(map(int, numbers))
#         job_ids = split_slurm_array(
#             path_to_task,
#             z_layers,
#             numbers,
#             partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
#             cores=1,
#             memory=32,
#             priority=priority,
#         )
#     else:
#         job_ids = submit_slurm_array(
#             path_to_task,
#             z_layers,
#             partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
#             cores=1,
#             memory=32,
#             priority=priority,
#         )
#     return job_ids


def submit_stats_job(input_dir, out_dir):
    path_to_task = os.path.join(jobs_folder, f"attenuation_corr_stats.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "get_stats.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-attenuation_corr_stats")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/attenuation_corr_stats_slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(input_dir if ' ' not in input_dir else f'"{input_dir}"')
        f.write(' ')
        f.write(out_dir if ' ' not in out_dir else f'"{out_dir}"')
        f.write(' ')
        f.write(str(resolution_level))
        f.write(' ')
        f.write(str(channel))
        f.write(' ')
        f.write(str(reference_z))
        f.write(' ')
        f.write(str(opening_radius))
        f.write('\n')

    # already_done = glob(os.path.join(out_dir, "*.npy"))
    # if len(already_done):
    #     print("Partially processed")
    #     print("Processed", len(already_done), "of", z_layers)
    #     files = os.listdir(save_folder)
    #     pattern = "_z(\d+)\.npy"
    #     numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.npy')]
    #     numbers = set(map(int, numbers))
    #     job_ids = split_slurm_array(
    #         path_to_task,
    #         z_layers,
    #         numbers,
    #         partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
    #         cores=1,
    #         memory=32,
    #         priority=priority,
    #     )
    # else:
    #     job_ids = submit_slurm_array(
    #         path_to_task,
    #         z_layers,
    #         partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
    #         cores=1,
    #         memory=32,
    #         priority=priority,
    #     )
    job_ids = submit_slurm_job(
        path_to_task,
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=1,
        memory=32,
        priority=priority
    )
    return job_ids


input_dir = sys.argv[1]
save_folder = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
username = sys.argv[5]
priority = sys.argv[6]
reference_z = sys.argv[7]
opening_radius = int(sys.argv[8])
experiment = sys.argv[9]

metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
z_layers = metadata['shape'][-3]

output_folder_sequence = Path(save_folder).parent
jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")

stats_folder = os.path.join(
    output_folder_sequence,
    'stats_ref'
)
try:
    os.makedirs(stats_folder)
except:
    pass

# min_max_folder_denoised = os.path.join(
#     output_folder_sequence,
#     f'min_max_denoised_model_{model}_diameter_{diameter}'
# )
# try:
#     os.makedirs(min_max_folder_denoised)
# except:
#     pass

base_input_dir = metadata['base_input_dir']

if not base_input_dir:
    base_input_dir = input_dir
base_metadata = json.load(open(os.path.join(base_input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
source_path = base_metadata['source']

stats_ref_npy = os.path.join(
    output_folder_sequence,
    'stats_ref',
    f"stats_reference_z_{reference_z}.npy"
)
if os.path.exists(stats_ref_npy):
    print("Stats file exists")
    stats_ref = np.load(stats_ref_npy)
    ref_mean, ref_std = stats_ref[0], stats_ref[1]
else:
    print("Calculating stats")
    submit_stats_job(input_dir, stats_folder)

    finished = os.path.exists(stats_ref_npy)
    while not finished:
        time.sleep(10)
        finished = os.path.exists(stats_ref_npy)

    stats_ref = np.load(stats_ref_npy)
    ref_mean, ref_std = stats_ref[0], stats_ref[1]

print("Reference Mean", ref_mean, "Reference Std", ref_std)

submit_correction_job_array()

finished_planes = len(glob(os.path.join(save_folder, '*.tif')))
while finished_planes < z_layers:
    print("finished", finished_planes, "of", z_layers)
    time.sleep(60)
    finished_planes = len(glob(os.path.join(save_folder, '*.tif')))

# min_max_denoised_npy = os.path.join(
#     output_folder_sequence,
#     f"min_max_{model}_diameter_{diameter}.npy"
# )
#
# if os.path.exists(min_max_denoised_npy):
#     min_max_denoised = np.load(min_max_denoised_npy)
#     denoised_stack_min, denoised_stack_max = min_max_denoised[0], min_max_denoised[1]
# else:
#     submit_min_max_job_array(save_folder, min_max_folder_denoised)  # double check that all min and max have been calculated
#     finished_planes = len(glob(os.path.join(min_max_folder_denoised, '*.npy')))
#     while finished_planes < z_layers:
#         print("finished", finished_planes, "of", z_layers)
#         time.sleep(10)
#         finished_planes = len(glob(os.path.join(min_max_folder_denoised, '*.npy')))
#
#     denoised_stack_min, denoised_stack_max = find_min_max(min_max_folder_denoised)  # TODO: store to a text file
#
#     np.save(
#         min_max_denoised_npy,
#         np.array([denoised_stack_min, denoised_stack_max])
#     )

# save_folder_uint = save_folder + "_uint"
# if not os.path.exists(save_folder_uint):
#     os.makedirs(save_folder_uint)

# submit_conversion_job_array()

finished_planes = len(glob(os.path.join(save_folder, '*.tif')))
while finished_planes < z_layers:
    print("finished", finished_planes, "of", z_layers)
    time.sleep(10)
    finished_planes = len(glob(os.path.join(save_folder, '*.tif')))

has_errors = parse_slurm_errors(jobs_folder)
print("Errors found:", has_errors)
