import subprocess

from analysis.settings import (
    SLURM_PARTITION_CPU, SLURM_PARTITION_GPU,
    PRIORITY_TO_NICE_MAP_COMPUTE, PRIORITY_TO_NICE_MAP_GPU
)


def submit_slurm_job(job_path, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0):
    if needs_gpu:
        nice = PRIORITY_TO_NICE_MAP_GPU[priority]
    else:
        nice = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
        nice = int(nice * (cores / 24.))

    command = [
        'sbatch',
        '-p', partition,
        '--gres=gpu:1' if needs_gpu and partition == SLURM_PARTITION_GPU else ''
        f'--mem={memory}Gb',
        f'-n{cores}',
        f'--nice={nice}',
        job_path
    ]
    subprocess.run(command)


def submit_slurm_array(job_path, number_of_tasks, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0):
    if needs_gpu:
        nice = PRIORITY_TO_NICE_MAP_GPU[priority]
    else:
        nice = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
        nice = int(nice * (cores / 24.))

    command = [
        'sbatch',
        f'--array=0-{number_of_tasks - 1}',
        '-p', partition,
        '--gres=gpu:1' if needs_gpu and partition == SLURM_PARTITION_GPU else ''
        f'--mem={memory}Gb',
        f'-n{cores}',
        f'--nice={nice}',
        job_path
    ]
    subprocess.run(command)


def submit_partial_slurm_array(job_path, array_start, array_end, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0):
    if needs_gpu:
        nice = PRIORITY_TO_NICE_MAP_GPU[priority]
    else:
        nice = PRIORITY_TO_NICE_MAP_COMPUTE[priority]
        nice = int(nice * (cores / 24.))

    command = [
        'sbatch',
        f'--array={array_start}-{array_end}',
        '-p', partition,
        '--gres=gpu:1' if needs_gpu and partition == SLURM_PARTITION_GPU else ''
        f'--mem={memory}Gb',
        f'-n{cores}',
        f'--nice={nice}',
        job_path
    ]
    subprocess.run(command)


def split_range_in_subranges(n, existing_outputs):
    existing_set = set(existing_outputs)
    subranges = []
    start = None

    for i in range(n):
    # for i in range(n + 1):
        if i not in existing_set:
            if start is None:
                start = i
        else:
            if start is not None:
                subranges.append((start, i - 1))
                start = None
    if start is not None:
        subranges.append((start, n))
    return subranges


def split_slurm_array(job_path, number_of_tasks, existing_outputs, partition=SLURM_PARTITION_CPU, cores=1, memory=8, needs_gpu=False, priority=0):
    subranges = split_range_in_subranges(number_of_tasks, existing_outputs)
    for start, end in subranges:
        submit_partial_slurm_array(
            job_path,
            start,
            end,
            partition=partition,
            cores=cores,
            memory=memory,
            needs_gpu=needs_gpu,
            priority=priority
        )
