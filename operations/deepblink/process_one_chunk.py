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


DEEPBLINK_MODEL_PATH = '/h20/CBI/Iana/src/deepblink/models/deepblink_particle.h5'
INFO_FILE_NAME = "dataset_info.json"

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
    command = [
        'sbatch',
        '-p', 'gpu',  # TODO use settings
        '--gres=gpu:1',
        '--mem=64Gb',
        '-n8',
        f'--nice=500', # TODO use settings
        path_to_task
    ]
    subprocess.run(command)


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


def extract_detect_deepblink_delete(number):
    def save_empty_napari_df():
        print(f"WARNING: saving empty napari DF for chunk # {number}")
        napari_csv_file_path = os.path.join(napari_folder, f"napari_chunk_{str(number).zfill(5)}.csv")
        df = pd.DataFrame()
        df.to_csv(napari_csv_file_path)

    print("Chunk number", number)
    try:
        extract_chunk_by_number(number)
    except:  # if impossible to extract, create an empty napari-compatible DF
        print(f"EXCEPTION: unable to extract chunk # {number}")
        print(traceback.format_exc())
        save_empty_napari_df()
        return
    print("extracted")
    detect_cells_deepblink_one_chunk(number)
    print("sent detection job")
    delete_extracted_chunk_by_number(number)
    print("Deleted")
    try:
        convert_one_csv_to_napari_format_by_number(number)
    except:
        print(f"EXCEPTION: unable to save to napari format chunk # {number}")
        save_empty_napari_df()
        return
    print("Converted to napari")


INPUT_DIR = sys.argv[1]  # TODO: this can be read directly from the JSON file
OUTPUT_DIR = sys.argv[2]
resolution_level = int(sys.argv[3])
signal_channel = int(sys.argv[4])
username = sys.argv[5]
chunk_number = int(sys.argv[6])

metadata = json.load(open(os.path.join(INPUT_DIR, f'.{INFO_FILE_NAME}'), 'r'))
source = metadata['source']

OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'deepblink', f'resolution_level_{resolution_level}', f'channel_{signal_channel}')
CHUNKS_FOLDER = os.path.join(OUTPUT_DIR, "deepblink_chunks")
jobs_folder = os.path.join(OUTPUT_DIR, "slurm_jobs")
detection_folder = os.path.join(OUTPUT_DIR, "detection")
napari_folder = os.path.join(OUTPUT_DIR, "detection_napari")

extract_detect_deepblink_delete(int(chunk_number))
