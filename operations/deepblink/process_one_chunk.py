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


DEEPBLINK_MODEL_PATH = settings.DEEPBLINK_MODEL_PATH
INFO_FILE_NAME = settings.INFO_FILE_NAME
PRIORITY_TO_NICE_MAP_GPU = settings.PRIORITY_TO_NICE_MAP_GPU
PRIORITY_TO_NICE_MAP_COMPUTE = settings.PRIORITY_TO_NICE_MAP_COMPUTE


def extract_chunk_from_imaris_by_number(number):
    print("Extracting chunks from the imaris file")
    chunk_file = os.path.join(CHUNKS_FOLDER, f"chunk_{str(number).zfill(5)}.tif")
    if os.path.exists(chunk_file):
        return
    chunk_indices_path = os.path.join(CHUNKS_FOLDER, 'chunk_indices.npy')
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

    chunk_file = os.path.join(CHUNKS_FOLDER, f"chunk_{str(number).zfill(5)}.tif")
    if os.path.exists(chunk_file):
        return

    # Get a sorted list of all image file paths
    image_files = sorted(
        [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(('.tif', '.tiff'))]
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

    chunk_indices_path = os.path.join(CHUNKS_FOLDER, 'chunk_indices.npy')
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)
    slices = chunk_indices[number]
    chunk = stacked_array[tuple(slices)]
    print(chunk.shape)
    chunk = chunk.compute()
    tifffile.imwrite(chunk_file, chunk)


def write_detection_task_for_slurm(chunk_number, output_path):
    input_file = os.path.join(CHUNKS_FOLDER, f"chunk_{str(chunk_number).zfill(5)}.tif")
    output_dir = os.path.join(OUTPUT_DIR, "detection")
    with open(output_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-deepblink-gpu")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate deepblink")
        f.write('\n')
        f.write(f'deepblink predict -m {DEEPBLINK_MODEL_PATH} -i')
        f.write(' ')
        f.write(input_file if ' ' not in input_file else f'"{input_file}"')
        f.write(' -o ')
        f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
        f.write('\n')


def submit_slurm_task_gpu(path_to_task):
    submit_slurm_job(
        path_to_task,
        partition=f'{settings.SLURM_PARTITION_GPU}',
        cores=8,
        memory=64,
        needs_gpu=True,
        priority=priority
    )
    # nice_value = PRIORITY_TO_NICE_MAP_GPU[priority]
    # command = [
    #     'sbatch',
    #     '-p', 'gpu',  # TODO use settings
    #     '--gres=gpu:1',
    #     '--mem=64Gb',
    #     '-n8',
    #     f'--nice={nice_value}',
    #     path_to_task
    # ]
    # subprocess.run(command)


def detect_cells_deepblink_one_chunk(chunk_number):
    detections_file_name = os.path.join(detection_folder, f"chunk_{str(chunk_number).zfill(5)}.csv")
    if os.path.exists(detections_file_name):
        print(f"Skipping chunk {chunk_number}")
        return
    print(f"Submitting gpu task for chunk {chunk_number}")
    task_path = os.path.join(jobs_folder, f"detect_chunk_{str(chunk_number).zfill(5)}.sh")
    write_detection_task_for_slurm(chunk_number, task_path)
    submit_slurm_task_gpu(task_path)


def delete_extracted_chunk_by_number(number):
    import time
    chunk_file = os.path.join(CHUNKS_FOLDER, f"chunk_{str(number).zfill(5)}.tif")
    detections_file_name = os.path.join(detection_folder, f"chunk_{str(number).zfill(5)}.csv")
    while not os.path.exists(detections_file_name):
        time.sleep(10)
    # os.remove(chunk_file)
    print('removed', chunk_file)


def convert_one_csv_to_napari_format_by_number(number):
    import pandas as pd
    csv_file = os.path.join(detection_folder, f"chunk_{str(number).zfill(5)}.csv")
    napari_csv_file_path = os.path.join(napari_folder, f"napari_{os.path.basename(csv_file)}")
    if os.path.exists(napari_csv_file_path):
        return
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError as e:
        print("Warning: ", e)
        df2 = pd.DataFrame()
        df2.to_csv(napari_csv_file_path)
        return
    df2 = pd.DataFrame()
    df2['index'] = list(range(df.shape[0]))
    zvals = df['z'].tolist()
    yvals = df['y [px]'].tolist()
    xvals = df['x [px]'].tolist()
    df2['axis-0'] = zvals
    df2['axis-1'] = xvals
    df2['axis-2'] = yvals
    df2.to_csv(napari_csv_file_path)


def extract_chunk_by_number(number):
    if source.endswith('.ims'):
        extract_chunk_from_imaris_by_number(number)
    else:
        extract_chunk_from_tiff_series_by_number(number)


def run_dbscan_on_chunk(chunk_number):
    path_to_task = os.path.join(jobs_folder, f"dbscan_{chunk_number}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "dbscan_one_chunk.py")
    input_file = os.path.join(napari_folder, f"napari_chunk_{str(chunk_number).zfill(5)}.csv")
    output_dir = os.path.join(OUTPUT_DIR, "dbscan")
    try:
        os.makedirs(output_dir)
    except:
        pass
    # nice_value = PRIORITY_TO_NICE_MAP_COMPUTE[priority]

    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-deepblink-dbscan")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate dbscan")
        f.write('\n')
        f.write(f'python {slurm_script} ')
        f.write(input_file if ' ' not in input_file else f'"{input_file}"')
        f.write(' ')
        f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
        f.write('\n')

    submit_slurm_job(
        path_to_task,
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=8,
        memory=32,
        priority=priority
    )
    # command = [
    #     'sbatch',
    #     '-p', 'compute,gpu',
    #     '--mem=32Gb',
    #     '-n8',
    #     f'--nice={nice_value}',
    #     path_to_task
    # ]
    # # print("command", command)
    # subprocess.run(command)


def extract_detect_deepblink_delete(number):
    def save_empty_napari_df():
        print(f"WARNING: saving empty napari DF for chunk # {number}")
        napari_csv_file_path = os.path.join(napari_folder, f"napari_chunk_{str(number).zfill(5)}.csv")
        df = pd.DataFrame()
        df.to_csv(napari_csv_file_path)

    print("Chunk number", number)
    chunk_file = os.path.join(CHUNKS_FOLDER, f"chunk_{str(number).zfill(5)}.tif")
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
    if not os.path.exists(detections_file_name):
        detect_cells_deepblink_one_chunk(number)
        print("sent detection job")
    else:
        print(f"Skipping chunk {number}")

    # delete_extracted_chunk_by_number(number)
    # print("Deleted")
    napari_file_name = os.path.join(napari_folder, f"napari_chunk_{str(number).zfill(5)}.csv")
    if not os.path.exists(napari_file_name):
        try:
            convert_one_csv_to_napari_format_by_number(number)
        except:
            print(f"EXCEPTION: unable to save to napari format chunk # {number}")
            save_empty_napari_df()
            return
        print("Converted to napari")
    dbscan_file_name = os.path.join(dbscan_folder, f"dbscan_napari_chunk_{str(number).zfill(5)}.csv")
    if with_dbscan and not os.path.exists(dbscan_file_name):
        run_dbscan_on_chunk(number)


INPUT_DIR = sys.argv[1]  # TODO: this can be read directly from the JSON file
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
signal_channel = int(sys.argv[4])
username = sys.argv[5]
with_dbscan = int(sys.argv[6])
priority = sys.argv[7]
chunk_number = int(sys.argv[8])

metadata = json.load(open(os.path.join(INPUT_DIR, f'.{INFO_FILE_NAME}'), 'r'))
source = metadata['source']

OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'deepblink', f'resolution_level_{resolution_level}', f'channel_{signal_channel}')
CHUNKS_FOLDER = os.path.join(OUTPUT_DIR, "deepblink_chunks")
jobs_folder = os.path.join(OUTPUT_DIR, "slurm_jobs")
detection_folder = os.path.join(OUTPUT_DIR, "detection")
napari_folder = os.path.join(OUTPUT_DIR, "detection_napari")
dbscan_folder = os.path.join(OUTPUT_DIR, "dbscan")

if with_dbscan:
    dbscan_file_name = os.path.join(dbscan_folder, f"dbscan_napari_chunk_{str(chunk_number).zfill(5)}.csv")
    if not os.path.exists(dbscan_file_name):
        extract_detect_deepblink_delete(int(chunk_number))
else:
    napari_file_name = os.path.join(napari_folder, f"napari_chunk_{str(chunk_number).zfill(5)}.csv")
    if not os.path.exists(napari_file_name):
        extract_detect_deepblink_delete(int(chunk_number))
