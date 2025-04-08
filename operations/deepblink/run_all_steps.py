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
from utils.slurm import split_slurm_array, submit_slurm_array

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


def submit_detection_cpu_slurm_array(number_of_chunks):
    # write slurm job
    path_to_task = os.path.join(jobs_folder, f"cpu_array_all_chunks.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "process_one_chunk.py")

    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-deepblink-cpu")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_deepblink_cpu_%A_%a.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script} ')
        f.write(INPUT_DIR)
        f.write(' ')
        f.write(OUTPUT_DIR)
        f.write(' ')
        f.write(f'{resolution_level} {signal_channel} {username} {with_dbscan} {priority} $SLURM_ARRAY_TASK_ID')
        f.write('\n')

    if with_dbscan:
        save_folder = dbscan_folder
    else:
        save_folder = napari_folder

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
            cores=12,
            memory=32,
            priority=priority
        )
    else:
        submit_slurm_array(
            path_to_task,
            number_of_chunks,
            partition=f'{settings.SLURM_PARTITION_CPU}',
            cores=12,
            memory=32,
            priority=priority
        )


def merge_df():
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(napari_folder, 'napari*.csv')))
    # print("CSV files", len(csv_files), csv_files[:3])
    origin_coords = np.load(os.path.join(chunks_folder, 'origin_coords.npy'), allow_pickle=True)
    for chunk_file in csv_files:
        current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
        print("Processing", current_chunk)
        chunk_df = pd.read_csv(chunk_file)
        if chunk_df.empty:
            continue
        chunk_df_corrected = pd.DataFrame()
        z_values = chunk_df[['axis-0']].to_numpy()
        y_values = chunk_df[['axis-1']].to_numpy()
        x_values = chunk_df[['axis-2']].to_numpy()
        z_values += origin_coords[current_chunk, 0]
        y_values += origin_coords[current_chunk, 1]
        x_values += origin_coords[current_chunk, 2]
        chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
        chunk_df_corrected['axis-0'] = z_values
        chunk_df_corrected['axis-1'] = y_values
        chunk_df_corrected['axis-2'] = x_values
        df = pd.concat([df, chunk_df_corrected])

    print("Saving df")
    df.to_csv(os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", 'merged_df.csv'))


def merge_dbscan_df():
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(dbscan_folder, 'dbscan*.csv')))
    origin_coords = np.load(os.path.join(chunks_folder, 'origin_coords.npy'), allow_pickle=True)
    for chunk_file in csv_files:
        current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
        print("Processing", current_chunk)
        chunk_df = pd.read_csv(chunk_file)
        if chunk_df.empty:
            continue
        chunk_df_corrected = pd.DataFrame()
        z_values = chunk_df[['axis-0']].to_numpy()
        y_values = chunk_df[['axis-1']].to_numpy()
        x_values = chunk_df[['axis-2']].to_numpy()
        z_values += origin_coords[current_chunk, 0]
        y_values += origin_coords[current_chunk, 1]
        x_values += origin_coords[current_chunk, 2]
        chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
        chunk_df_corrected['axis-0'] = z_values
        chunk_df_corrected['axis-1'] = y_values
        chunk_df_corrected['axis-2'] = x_values
        df = pd.concat([df, chunk_df_corrected])

    print("Saving df")
    df.to_csv(os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", 'merged_dbscan_df.csv'))


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
signal_channel = int(sys.argv[4])
username = sys.argv[5]
with_dbscan = int(sys.argv[6])
priority = sys.argv[7]

metadata = json.load(open(os.path.join(INPUT_DIR, f'.{settings.INFO_FILE_NAME}'), 'r'))
output_operation_folder = os.path.join(OUTPUT_DIR, 'deepblink')
napari_folder = os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", "detection_napari")
chunks_folder = os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", "deepblink_chunks")
jobs_folder = os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", "slurm_jobs")
dbscan_folder = os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", "dbscan")

number_of_chunks = get_chunking()
print("number of chunks", number_of_chunks)
submit_detection_cpu_slurm_array(number_of_chunks)
# wait for all GPU jobs to finish
chunks_done = len(glob(os.path.join(napari_folder, "napari_chunk_*.csv")))
while chunks_done < number_of_chunks:
    print(f"Chunks done: {chunks_done} of {number_of_chunks}")
    time.sleep(120)
    chunks_done = len(glob(os.path.join(napari_folder, "napari_chunk_*.csv")))

# merge the dataframes with points for all chunks
merged_csv = os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", 'merged_df.csv')
if not os.path.exists(merged_csv):
    merge_df()

if with_dbscan:
    chunks_done = len(glob(os.path.join(dbscan_folder, "dbscan_napari_chunk_*.csv")))
    while chunks_done < number_of_chunks:
        print(f"Chunks done: {chunks_done} of {number_of_chunks}")
        time.sleep(120)
        chunks_done = len(glob(os.path.join(dbscan_folder, "dbscan_napari_chunk_*.csv")))
    # merge the DBSCAN dataframes with points for all chunks
    merged_dbscan_csv = os.path.join(output_operation_folder, f"resolution_level_{resolution_level}", f"channel_{signal_channel}", 'merged_dbscan_df.csv')
    if not os.path.exists(merged_dbscan_csv):
        merge_dbscan_df()

print("All done!")
