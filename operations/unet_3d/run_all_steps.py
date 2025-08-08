import json
import os
import re
import subprocess
import sys
import time
from glob import glob

import numpy as np
import pandas as pd

from pathlib import Path
this_script = Path(__file__)
deepblink_folder = this_script.parent
operations_folder = deepblink_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import split_slurm_array, submit_slurm_array, parse_slurm_errors
from utils.chunks import extract_chunk_from_imaris_by_number, extract_chunk_from_tiff_series_by_number

os.umask(settings.UMASK)

CHUNK_SIZE = settings.DEEPBLINK_CHUNK_SIZE


def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    """
    Get coordinates of each chunk origin.
    """
    coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
    coords = np.empty(coords_shape, dtype=np.uint16)
    print(" coords shape", coords.shape)
    for z in range(coords.shape[0]):
        for y in range(coords.shape[1]):
            for x in range(coords.shape[2]):
                coords[z, y, x, :] = np.array((
                    z * chunk_size[0],
                    y * chunk_size[1],
                    x * chunk_size[2]
                ))
    coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
    print("final coords shape", coords.shape)
    return coords


def get_chunk_indices(origin_coords, chunk_size):
    indices = []
    for origin in list(origin_coords):
        indices.append([
            slice(origin[0], origin[0] + chunk_size[0], 1),
            slice(origin[1], origin[1] + chunk_size[1], 1),
            slice(origin[2], origin[2] + chunk_size[2], 1)
        ])
    return indices


def get_chunking():
    tiff_stack_shape = metadata['shape']
    ratios = (np.array(tiff_stack_shape) / np.array(CHUNK_SIZE)).astype('int') + 1
    patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
    print("patchify_chunks_shape", patchify_chunks_shape)
    origin_coords = get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
    chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)
    print("Total chunks", len(chunk_indices))
    np.save(os.path.join(chunks_folder, 'origin_coords.npy'), origin_coords)
    np.save(os.path.join(chunks_folder, 'chunk_indices.npy'), chunk_indices)
    return len(chunk_indices)


def extract_first_chunk():
    source = metadata['source']
    if source.endswith('.ims'):
        extract_chunk_from_imaris_by_number(0, chunks_folder)
    else:
        extract_chunk_from_tiff_series_by_number(0, chunks_folder)


def submit_detection_cpu_slurm_array(number_of_chunks):
    # write slurm job
    path_to_task = os.path.join(jobs_folder, f"cpu_array_all_chunks.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "process_one_chunk.py")

    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-3d-unet-cpu")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/logs_process_one_chunk/slurm-3d-unet-cpu_%A_%a.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script} ')
        f.write(input_dir)
        f.write(' ')
        f.write(output_folder_sequence)
        f.write(' ')
        f.write(chunks_folder)
        f.write(' ')
        f.write(f'{resolution_level} {channel} $SLURM_ARRAY_TASK_ID {username} {model} {priority}')
        f.write('\n')

    already_done = glob(os.path.join(save_folder, "*.csv"))

    if len(already_done):
        print("Partially processed")
        print("Processed", len(already_done), "of", number_of_chunks)
        files = os.listdir(save_folder)
        pattern = "_chunk_(\d+)\.csv"
        numbers = [re.findall(pattern, x)[0] for x in files]
        numbers = set(map(int, numbers))
        split_slurm_array(
            path_to_task,
            number_of_chunks,
            numbers,
            partition=','.join([settings.SLURM_PARTITION_CPU]),
            cores=1,
            memory=8,
            priority=priority
        )
    else:
        submit_slurm_array(
            path_to_task,
            number_of_chunks,
            partition=f'{settings.SLURM_PARTITION_CPU}',
            cores=1,
            memory=8,
            priority=priority
        )


input_dir = sys.argv[1]
output_folder_sequence = sys.argv[2]
chunks_folder = sys.argv[3]
resolution_level = int(sys.argv[4])
signal_channel = int(sys.argv[5])
username = sys.argv[6]
model = sys.argv[7]
priority = sys.argv[8]

metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
napari_folder = os.path.join(output_folder_sequence, "detection_napari")
# chunks_folder = os.path.join(output_folder_sequence, "deepblink_chunks")
jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
dbscan_folder = os.path.join(output_folder_sequence, "dbscan")

number_of_chunks = get_chunking()
print("number of chunks", number_of_chunks)
extract_first_chunk()  # a reference chunk, considered having all noise
submit_detection_cpu_slurm_array(number_of_chunks)
# wait for all GPU jobs to finish
chunks_done = len(glob(os.path.join(napari_folder, "napari_chunk_*.csv")))
while chunks_done < number_of_chunks:
    print(f"Chunks done: {chunks_done} of {number_of_chunks}")
    time.sleep(120)
    chunks_done = len(glob(os.path.join(napari_folder, "napari_chunk_*.csv")))

# merge the dataframes with points for all chunks
merged_csv = os.path.join(output_folder_sequence, 'merged_df.csv')
if not os.path.exists(merged_csv):
    merge_df()

if with_dbscan:
    chunks_done = len(glob(os.path.join(dbscan_folder, "dbscan_napari_chunk_*.csv")))
    while chunks_done < number_of_chunks:
        print(f"Chunks done: {chunks_done} of {number_of_chunks}")
        time.sleep(120)
        chunks_done = len(glob(os.path.join(dbscan_folder, "dbscan_napari_chunk_*.csv")))
    # merge the DBSCAN dataframes with points for all chunks
    merged_dbscan_csv = os.path.join(output_folder_sequence, 'merged_dbscan_df.csv')
    if not os.path.exists(merged_dbscan_csv):
        merge_dbscan_df()

has_errors = parse_slurm_errors(jobs_folder)
print("Errors found:", has_errors)
print("All done!")
