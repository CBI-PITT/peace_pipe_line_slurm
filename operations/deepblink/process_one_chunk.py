import json
import os
import subprocess
import sys
import traceback
from glob import glob

import numpy as np
import pandas as pd
import dask.array as da
from imaris_ims_file_reader import ims
import tifffile

from pathlib import Path
this_script = Path(__file__)
deepblink_folder = this_script.parent
operations_folder = deepblink_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import submit_slurm_job

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def extract_chunk_from_imaris_by_number(number):
    print("Extracting chunks from the imaris file")
    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    if os.path.exists(chunk_file):
        return
    chunk_indices_path = os.path.join(chunks_folder, 'chunk_indices.npy')
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)
    slices = chunk_indices[number]
    ims_file = ims(metadata['source'], ResolutionLevelLock=resolution_level)
    ims_file_dask = da.array(ims_file)
    tiffstack = ims_file_dask[0, signal_channel, :, :, :]
    chunk = tiffstack[tuple(slices)]
    print(chunk.shape)
    chunk = chunk.compute()
    tifffile.imwrite(chunk_file, chunk)


def extract_chunk_from_tiff_series_by_number(number):
    print("Extracting chunks from the tiff series")

    from dask import array as da
    from dask import delayed

    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    if os.path.exists(chunk_file):
        return

    # Get a sorted list of all image file paths
    image_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tif', '.tiff'))]
    )
    z, y, x = metadata['shape']

    # Function to load a single image
    @delayed
    def read_image(file_path):
        return tifffile.imread(file_path)

    # Create a Dask array
    lazy_arrays = [da.from_delayed(read_image(f), shape=(y, x), dtype='uint16') for f in image_files]
    stacked_array = da.stack(lazy_arrays, axis=0)  # Stack along a new axis to get shape (z, y, x)

    # Check the resulting array shape and type
    print(stacked_array.shape)

    chunk_indices_path = os.path.join(chunks_folder, 'chunk_indices.npy')
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)
    slices = chunk_indices[number]
    chunk = stacked_array[tuple(slices)]
    print(chunk.shape)
    chunk = chunk.compute()
    tifffile.imwrite(chunk_file, chunk)


def write_detection_task_for_slurm(chunk_number, output_path):
    input_file = os.path.join(chunks_folder, f"chunk_{str(chunk_number).zfill(5)}.tif")
    output_dir = os.path.join(output_folder_sequence, "detection")
    with open(output_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-deepblink-gpu")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/logs_deepblink/slurm_deepblink_chunk_{str(chunk_number).zfill(5)}.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate deepblink")
        f.write('\n')
        f.write(f'deepblink predict -m {settings.DEEPBLINK_MODEL_PATH} -i')
        f.write(' ')
        f.write(input_file if ' ' not in input_file else f'"{input_file}"')
        f.write(' -o ')
        f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
        f.write('\n')


def submit_slurm_task_gpu(path_to_task):
    job_ids = submit_slurm_job(
        path_to_task,
        partition=f'{settings.SLURM_PARTITION_GPU}',
        cores=1,
        memory=64,
        needs_gpu=True,
        priority=priority
    )
    return job_ids


def detect_cells_deepblink_one_chunk(chunk_number):
    detections_file_name = os.path.join(detection_folder, f"chunk_{str(chunk_number).zfill(5)}.csv")
    if os.path.exists(detections_file_name):
        print(f"Skipping chunk {chunk_number}")
        return
    print(f"Submitting gpu task for chunk {chunk_number}")
    task_path = os.path.join(jobs_folder, f"detect_chunk_{str(chunk_number).zfill(5)}.sh")
    write_detection_task_for_slurm(chunk_number, task_path)
    job_ids = submit_slurm_task_gpu(task_path)
    return job_ids


def delete_extracted_chunk_by_number(number):
    import time
    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    detections_file_name = os.path.join(detection_folder, f"chunk_{str(number).zfill(5)}.csv")
    while not os.path.exists(detections_file_name):
        time.sleep(10)
    # os.remove(chunk_file)
    print('removed', chunk_file)


def convert_one_csv_to_napari_format_by_number(number, prerequisites=None):
    path_to_task = os.path.join(jobs_folder, f"napari_{chunk_number}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "napari_one_chunk.py")

    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-deepblink-napari")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/logs_napari/slurm_napari_chunk_{str(number).zfill(5)}.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(str(number))
        f.write(' ')
        f.write(detection_folder if ' ' not in detection_folder else f'"{detection_folder}"')
        f.write(' ')
        f.write(napari_folder if ' ' not in napari_folder else f'"{napari_folder}"')
        f.write('\n')

    extra_args = {}
    if prerequisites:
        extra_args['--depend'] = f'afterok:{":".join(list(map(str, prerequisites)))}'
        extra_args['--kill-on-invalid-dep'] = 'yes'

    job_ids = submit_slurm_job(
        path_to_task,
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=1,
        memory=8,
        priority=priority,
        extra_args=extra_args
    )
    return job_ids


def extract_chunk_by_number(number):
    if source.endswith('.ims'):
        extract_chunk_from_imaris_by_number(number)
    else:
        extract_chunk_from_tiff_series_by_number(number)


def run_dbscan_on_chunk(chunk_number, prerequisites=None):
    path_to_task = os.path.join(jobs_folder, f"dbscan_{chunk_number}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "dbscan_one_chunk.py")
    input_file = os.path.join(napari_folder, f"napari_chunk_{str(chunk_number).zfill(5)}.csv")
    output_dir = os.path.join(output_folder_sequence, "dbscan")
    try:
        os.makedirs(output_dir)
    except:
        pass

    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-deepblink-dbscan")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/logs_dbscan/slurm_dbscan_chunk_{str(chunk_number).zfill(5)}.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate dbscan")
        f.write('\n')
        f.write(f'python {slurm_script} ')
        f.write(input_file if ' ' not in input_file else f'"{input_file}"')
        f.write(' ')
        f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
        f.write('\n')

    extra_args = {}
    if prerequisites:
        extra_args['--depend'] = f'afterok:{":".join(list(map(str, prerequisites)))}'
        extra_args['--kill-on-invalid-dep'] = 'yes'

    submit_slurm_job(
        path_to_task,
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=1,
        memory=8,
        priority=priority,
        extra_args=extra_args
    )


def extract_detect_deepblink_delete(number):
    def save_empty_napari_df():
        print(f"WARNING: saving empty napari DF for chunk # {number}")
        napari_csv_file_path = os.path.join(napari_folder, f"napari_chunk_{str(number).zfill(5)}.csv")
        df = pd.DataFrame()
        df.to_csv(napari_csv_file_path)

    print("Chunk number", number)
    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    if not os.path.exists(chunk_file):
        try:
            extract_chunk_by_number(number)
        except:  # if impossible to extract, create an empty napari-compatible DF
            print(f"EXCEPTION: unable to extract chunk # {number}")
            print(traceback.format_exc())
            save_empty_napari_df()
            # return
        print("extracted")
    detections_file_name = os.path.join(detection_folder, f"chunk_{str(number).zfill(5)}.csv")
    detection_job_ids = []
    if not os.path.exists(detections_file_name):
        detection_job_ids = detect_cells_deepblink_one_chunk(number)
        print("sent detection job", detection_job_ids)
    else:
        print(f"Skipping chunk {number}")

    napari_file_name = os.path.join(napari_folder, f"napari_chunk_{str(number).zfill(5)}.csv")
    napari_job_ids = []
    if not os.path.exists(napari_file_name):
        napari_job_ids = convert_one_csv_to_napari_format_by_number(number, prerequisites=detection_job_ids)
        print("sent napari job", napari_job_ids)

    dbscan_file_name = os.path.join(dbscan_folder, f"dbscan_napari_chunk_{str(number).zfill(5)}.csv")
    if with_dbscan and not os.path.exists(dbscan_file_name):
        run_dbscan_on_chunk(number, prerequisites=napari_job_ids)


input_dir = sys.argv[1]  # TODO: this can be read directly from the JSON file
output_folder_sequence = sys.argv[2]
resolution_level = int(sys.argv[3])
signal_channel = int(sys.argv[4])
username = sys.argv[5]
with_dbscan = int(sys.argv[6])
priority = sys.argv[7]
chunk_number = int(sys.argv[8])

print("CHUNK", chunk_number)

metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
source = metadata['source']

chunks_folder = os.path.join(output_folder_sequence, "deepblink_chunks")
jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
detection_folder = os.path.join(output_folder_sequence, "detection")
napari_folder = os.path.join(output_folder_sequence, "detection_napari")
dbscan_folder = os.path.join(output_folder_sequence, "dbscan")

if with_dbscan:
    dbscan_file_name = os.path.join(dbscan_folder, f"dbscan_napari_chunk_{str(chunk_number).zfill(5)}.csv")
    if not os.path.exists(dbscan_file_name):
        extract_detect_deepblink_delete(int(chunk_number))
else:
    napari_file_name = os.path.join(napari_folder, f"napari_chunk_{str(chunk_number).zfill(5)}.csv")
    if not os.path.exists(napari_file_name):
        extract_detect_deepblink_delete(int(chunk_number))
