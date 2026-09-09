import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

this_script = Path(__file__)
operation_folder = this_script.parent
operations_folder = operation_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import parse_slurm_errors, submit_slurm_indices
from utils.z_range import (
    existing_z_indices,
    get_available_z_range,
    normalize_z_range,
    z_range_suffix,
)

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def min_max_file(folder, resolution_level, channel, z):
    return os.path.join(
        folder,
        f"r{resolution_level:02d}_t00_c{channel:02d}_z{z:04d}.npy"
    )


def find_min_max(folder, indices, resolution_level, channel):
    values = [
        np.load(min_max_file(folder, resolution_level, channel, z))
        for z in indices
    ]
    stack_min = min(value[0] for value in values)
    stack_max = max(value[1] for value in values)
    return stack_min, stack_max


def wait_for_indices(folder, indices, extension, poll_seconds):
    expected = set(indices)
    while True:
        completed = existing_z_indices(folder, extension=extension)
        missing = expected - completed
        if not missing:
            return
        print("finished", len(expected) - len(missing), "of", len(expected))
        time.sleep(poll_seconds)


def require_submitted_or_complete(job_ids, folder, indices, extension):
    missing = set(indices) - existing_z_indices(folder, extension=extension)
    if missing and not job_ids:
        raise RuntimeError(
            f"No SLURM jobs were submitted for missing z indices: {sorted(missing)[:10]}"
        )


def submit_denoising_job_array(
    input_dir,
    save_folder,
    min_max_folder_denoised,
    jobs_folder,
    selected_indices,
    resolution_level,
    channel,
    username,
    priority,
    model,
    diameter,
    stack_min,
    stack_max,
):
    path_to_task = os.path.join(
        jobs_folder,
        f"cellpose_denoise_rl{resolution_level}_c{channel}.sh"
    )
    slurm_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "do_denoising.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f"#SBATCH -J {username}-denoise-cellpose\n")
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out\n\n")
        f.write("source /h20/home/lab/miniconda3/bin/activate cellpose\n")
        args = [
            input_dir,
            save_folder,
            str(resolution_level),
            str(channel),
            '$SLURM_ARRAY_TASK_ID',
            str(model),
            str(diameter),
            str(stack_min),
            str(stack_max),
            min_max_folder_denoised,
        ]
        quoted_args = [f'"{arg}"' if ' ' in arg else arg for arg in args]
        f.write(f"python {slurm_script} {' '.join(quoted_args)}\n")

    return submit_slurm_indices(
        path_to_task,
        selected_indices,
        existing_outputs=existing_z_indices(save_folder),
        partition=','.join([settings.SLURM_PARTITION_GPU, settings.SLURM_PARTITION_EXTREME]),
        cores=1,
        memory=32,
        needs_gpu=True,
        priority=priority,
    )


def submit_conversion_job_array(
    save_folder,
    save_folder_uint,
    jobs_folder,
    selected_indices,
    resolution_level,
    channel,
    username,
    priority,
    denoised_stack_min,
    denoised_stack_max,
):
    path_to_task = os.path.join(
        jobs_folder,
        f"cellpose_convert_rl{resolution_level}_c{channel}.sh"
    )
    slurm_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_as_uint.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f"#SBATCH -J {username}-denoise-cellpose-convert\n")
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out\n\n")
        f.write("source /h20/home/lab/miniconda3/bin/activate peace\n")
        args = [
            save_folder,
            save_folder_uint,
            str(resolution_level),
            str(channel),
            '$SLURM_ARRAY_TASK_ID',
            str(denoised_stack_min),
            str(denoised_stack_max),
        ]
        quoted_args = [f'"{arg}"' if ' ' in arg else arg for arg in args]
        f.write(f"python {slurm_script} {' '.join(quoted_args)}\n")

    return submit_slurm_indices(
        path_to_task,
        selected_indices,
        existing_outputs=existing_z_indices(save_folder_uint),
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=1,
        memory=32,
        priority=priority,
    )


def submit_min_max_job_array(
    input_dir,
    out_dir,
    jobs_folder,
    indices,
    resolution_level,
    channel,
    username,
    priority,
):
    stage_name = os.path.basename(out_dir)
    path_to_task = os.path.join(
        jobs_folder,
        f"cellpose_min_max_{stage_name}_rl{resolution_level}_c{channel}.sh"
    )
    slurm_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "get_min_max.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write(f"#SBATCH -J {username}-denoise-cellpose-min-max\n")
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out\n\n")
        f.write("source /h20/home/lab/miniconda3/bin/activate peace\n")
        args = [
            input_dir,
            out_dir,
            str(resolution_level),
            str(channel),
            '$SLURM_ARRAY_TASK_ID',
        ]
        quoted_args = [f'"{arg}"' if ' ' in arg else arg for arg in args]
        f.write(f"python {slurm_script} {' '.join(quoted_args)}\n")

    return submit_slurm_indices(
        path_to_task,
        indices,
        existing_outputs=existing_z_indices(out_dir, extension='npy'),
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=1,
        memory=32,
        priority=priority,
    )


def move_stage_to_trash(path, username, experiment):
    if not os.path.exists(path):
        return
    trash_location = os.path.join(
        settings.TRASH_FOLDER,
        username,
        experiment,
        os.path.basename(path)
    )
    os.makedirs(trash_location, exist_ok=True)
    shutil.move(path, trash_location)


def main():
    input_dir = sys.argv[1]
    save_folder = sys.argv[2]
    save_folder_uint = sys.argv[3]
    resolution_level = int(sys.argv[4])
    channel = int(sys.argv[5])
    username = sys.argv[6]
    priority = sys.argv[7]
    model = sys.argv[8]
    diameter = int(sys.argv[9])
    experiment = sys.argv[10]
    requested_z_start = int(sys.argv[11])
    requested_z_end = int(sys.argv[12])

    with open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r') as f:
        metadata = json.load(f)
    selection = normalize_z_range(metadata, requested_z_start, requested_z_end)
    selected_indices = list(range(selection['start'], selection['end']))
    available_start, available_end = get_available_z_range(metadata)
    raw_indices = list(range(available_start, available_end))
    suffix = z_range_suffix(selection)
    total_z = int(metadata['shape'][-3])
    if available_start == 0 and available_end == total_z:
        raw_suffix = ''
    else:
        raw_suffix = f'_z{available_start}-{available_end}'

    output_folder_sequence = Path(save_folder).parent
    jobs_folder = os.path.join(output_folder_sequence, f"slurm_jobs{suffix}")
    os.makedirs(jobs_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder_uint, exist_ok=True)

    if set(selected_indices).issubset(existing_z_indices(save_folder_uint)):
        print("All selected uint outputs already exist")
        print("Errors found:", parse_slurm_errors(jobs_folder))
        return

    min_max_folder = os.path.join(output_folder_sequence, f'min_max_raw{raw_suffix}')
    min_max_folder_denoised = os.path.join(
        output_folder_sequence,
        f'min_max_denoised_model_{model}_diameter_{diameter}{suffix}'
    )
    os.makedirs(min_max_folder, exist_ok=True)
    os.makedirs(min_max_folder_denoised, exist_ok=True)

    min_max_raw_npy = os.path.join(output_folder_sequence, f"min_max_raw{raw_suffix}.npy")
    if os.path.exists(min_max_raw_npy):
        min_max_raw = np.load(min_max_raw_npy)
        stack_min, stack_max = min_max_raw[0], min_max_raw[1]
    else:
        print("Calculating raw min and max")
        raw_jobs = submit_min_max_job_array(
            input_dir,
            min_max_folder,
            jobs_folder,
            raw_indices,
            resolution_level,
            channel,
            username,
            priority,
        )
        require_submitted_or_complete(raw_jobs, min_max_folder, raw_indices, 'npy')
        wait_for_indices(min_max_folder, raw_indices, 'npy', 10)
        stack_min, stack_max = find_min_max(
            min_max_folder,
            raw_indices,
            resolution_level,
            channel,
        )
        np.save(min_max_raw_npy, np.array([stack_min, stack_max]))

    print("Raw stack_min", stack_min)
    print("Raw stack_max", stack_max)

    denoise_jobs = submit_denoising_job_array(
        input_dir,
        save_folder,
        min_max_folder_denoised,
        jobs_folder,
        selected_indices,
        resolution_level,
        channel,
        username,
        priority,
        model,
        diameter,
        stack_min,
        stack_max,
    )
    require_submitted_or_complete(denoise_jobs, save_folder, selected_indices, 'tif')
    wait_for_indices(save_folder, selected_indices, 'tif', 60)

    min_max_denoised_npy = os.path.join(
        output_folder_sequence,
        f"min_max_{model}_diameter_{diameter}{suffix}.npy"
    )
    if os.path.exists(min_max_denoised_npy):
        min_max_denoised = np.load(min_max_denoised_npy)
        denoised_stack_min, denoised_stack_max = min_max_denoised[0], min_max_denoised[1]
    else:
        denoised_stat_jobs = submit_min_max_job_array(
            save_folder,
            min_max_folder_denoised,
            jobs_folder,
            selected_indices,
            resolution_level,
            channel,
            username,
            priority,
        )
        require_submitted_or_complete(
            denoised_stat_jobs,
            min_max_folder_denoised,
            selected_indices,
            'npy'
        )
        wait_for_indices(min_max_folder_denoised, selected_indices, 'npy', 10)
        denoised_stack_min, denoised_stack_max = find_min_max(
            min_max_folder_denoised,
            selected_indices,
            resolution_level,
            channel,
        )
        np.save(
            min_max_denoised_npy,
            np.array([denoised_stack_min, denoised_stack_max])
        )

    conversion_jobs = submit_conversion_job_array(
        save_folder,
        save_folder_uint,
        jobs_folder,
        selected_indices,
        resolution_level,
        channel,
        username,
        priority,
        denoised_stack_min,
        denoised_stack_max,
    )
    require_submitted_or_complete(conversion_jobs, save_folder_uint, selected_indices, 'tif')
    wait_for_indices(save_folder_uint, selected_indices, 'tif', 10)

    move_stage_to_trash(save_folder, username, experiment)
    move_stage_to_trash(min_max_folder, username, experiment)
    move_stage_to_trash(min_max_folder_denoised, username, experiment)

    print("Errors found:", parse_slurm_errors(jobs_folder))


if __name__ == '__main__':
    main()
