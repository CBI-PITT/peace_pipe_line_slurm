import os
import re
import shutil
import subprocess

from analysis.settings import (
    SLURM_PARTITION_CPU, GPU_ENABLED_PARTITIONS,
    PRIORITY_TO_NICE_MAP_COMPUTE, PRIORITY_TO_NICE_MAP_GPU, SLURM_PARTITION_EXTREME
)


def get_job_number_from_slurm_out(output):
    return int(output.stdout[20:-1])


def submit_slurm_job(job_path, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0, extra_args=None):
    if needs_gpu:
        nice = PRIORITY_TO_NICE_MAP_GPU[priority]
    else:
        nice = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
        nice = int(nice * (cores / 24.))

    command = ['sbatch', '-p', partition]
    if needs_gpu and partition in GPU_ENABLED_PARTITIONS:
        command.append('--gres=gpu:1')
    command.extend([
        f'--mem={memory}Gb',
        f'-n{cores}',
        f'--nice={nice}',
    ])
    if extra_args and type(extra_args) is dict:
        for k, v in extra_args.items():
            command.append(f"{k}={v}")
    command.append(job_path)
    output = subprocess.run(command, capture_output=True, text=True)
    try:
        job_number = get_job_number_from_slurm_out(output)
        return [job_number]
    except:
        return []


def submit_slurm_array(job_path, number_of_tasks, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0, extra_args=None):
    if needs_gpu:
        nice = PRIORITY_TO_NICE_MAP_GPU[priority]
    else:
        nice = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
        nice = int(nice * (cores / 24.))

    command = [
        'sbatch',
        f'--array=0-{number_of_tasks - 1}',
        '-p', partition,
    ]
    if needs_gpu and partition in GPU_ENABLED_PARTITIONS:
        command.append('--gres=gpu:1')
    command.extend([
        f'--mem={memory}Gb',
        f'-n{cores}',
        f'--nice={nice}',
    ])
    if extra_args and type(extra_args) is dict:
        for k, v in extra_args.items():
            command.append(f"{k}={v}")
    command.append(job_path)
    output = subprocess.run(command, capture_output=True, text=True)
    try:
        job_number = get_job_number_from_slurm_out(output)
        return [job_number]
    except:
        return []


def submit_partial_slurm_array(job_path, array_start, array_end, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0, extra_args=None):
    if needs_gpu:
        nice = PRIORITY_TO_NICE_MAP_GPU[priority]
    else:
        nice = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
        nice = int(nice * (cores / 24.))

    command = [
        'sbatch',
        f'--array={array_start}-{array_end}',
        '-p', partition,
    ]
    if needs_gpu and partition in GPU_ENABLED_PARTITIONS:
        command.append('--gres=gpu:1')
    command.extend([
        f'--mem={memory}Gb',
        f'-n{cores}',
        f'--nice={nice}',
    ])
    if extra_args and type(extra_args) is dict:
        for k, v in extra_args.items():
            command.append(f"{k}={v}")
    command.append(job_path)
    output = subprocess.run(command, capture_output=True, text=True)
    try:
        job_number = get_job_number_from_slurm_out(output)
        return job_number
    except:
        return None


def split_indices_in_subranges(indices):
    indices = sorted(set(indices))
    if not indices:
        return []

    subranges = []
    start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            subranges.append((start, previous))
            start = index
        previous = index
    subranges.append((start, previous))
    return subranges


def submit_slurm_indices(
    job_path,
    task_indices,
    existing_outputs=None,
    partition=SLURM_PARTITION_CPU,
    cores=1,
    memory=8,
    needs_gpu=False,
    priority=0,
    extra_args=None,
):
    completed = set(existing_outputs or [])
    missing = set(task_indices) - completed
    job_ids = []

    for start, end in split_indices_in_subranges(missing):
        job_id = submit_partial_slurm_array(
            job_path,
            start,
            end,
            partition=partition,
            cores=cores,
            memory=memory,
            needs_gpu=needs_gpu,
            priority=priority,
            extra_args=extra_args,
        )
        if not job_id:
            raise RuntimeError(
                f"Unable to submit SLURM array range {start}-{end} for {job_path}"
            )
        job_ids.append(job_id)
    return job_ids


def split_range_in_subranges(n, existing_outputs):
    existing_set = set(existing_outputs)
    missing = [index for index in range(n) if index not in existing_set]
    return split_indices_in_subranges(missing)


def split_slurm_array(job_path, number_of_tasks, existing_outputs, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0, extra_args=None):
    return submit_slurm_indices(
        job_path,
        range(number_of_tasks),
        existing_outputs=existing_outputs,
        partition=partition,
        cores=cores,
        memory=memory,
        needs_gpu=needs_gpu,
        priority=priority,
        extra_args=extra_args,
    )


def parse_slurm_errors(logs_folder):
    """
    Recursively parses SLURM .out log files in the given folder and its subfolders.
    Moves any files containing errors into an 'errors' subfolder at the top level.

    Parameters:
    logs_folder (str): Path to the folder containing SLURM .out log files.
    """
    if not os.path.isdir(logs_folder):
        raise ValueError(f"Provided path '{logs_folder}' is not a valid directory.")

    errors_folder = os.path.join(logs_folder, "errors")
    os.makedirs(errors_folder, exist_ok=True)

    # Error indicators: python tracebacks, segfaults, common shell error patterns
    error_patterns = [
        r"Traceback \(most recent call last\)",
        r"Error:",
        r"Exception:",
        r"Segmentation fault",
        r"command not found",
        r"No such file or directory",
        r"Permission denied",
        r"ModuleNotFoundError",
        r"FileNotFoundError",
        r"ImportError",
        r"RuntimeError",
        r"ValueError",
        r"OSError",
    ]
    error_regex = re.compile("|".join(error_patterns), re.IGNORECASE)
    error_flag = False
    for root, _, files in os.walk(logs_folder):
        for filename in files:
            if filename.endswith(".out"):
                file_path = os.path.join(root, filename)

                # Avoid reprocessing files already in the errors folder
                if os.path.commonpath([file_path, errors_folder]) == errors_folder:
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        contents = f.read()
                        if error_regex.search(contents):
                            error_flag = True
                            # Ensure unique filename if subfolders have duplicate names
                            rel_path = os.path.relpath(file_path, logs_folder)
                            flat_name = rel_path.replace(os.sep, "__")
                            dest_path = os.path.join(errors_folder, flat_name)
                            shutil.move(file_path, dest_path)
                except Exception as e:
                    print(f"Could not read {file_path}: {e}")

    return error_flag
