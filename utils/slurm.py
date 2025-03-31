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

    print("============nice===========", nice)
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

    print("============nice===========", nice)
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
