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
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import submit_slurm_job

os.umask(settings.UMASK)


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
    output_dir = segmentation_folder
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "do_unet.py")
    with open(output_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-3d-unet-gpu")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate unet_3d")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(input_file if ' ' not in input_file else f'"{input_file}"')
        f.write(' ')
        f.write(output_dir if ' ' not in output_dir else f'"{output_dir}"')
        f.write(' ')
        f.write(str(resolution_level))
        f.write(' ')
        f.write(str(signal_channel))
        f.write(' ')
        f.write(str(model))
        f.write('\n')


def submit_slurm_task_gpu(path_to_task):
    submit_slurm_job(
        path_to_task,
        partition=settings.SLURM_PARTITION_GPU,
        needs_gpu=True,
        cores=8,
        memory=64,
        priority=priority
    )


def segment_one_chunk(chunk_number):
    detections_file_name = os.path.join(segmentation_folder, f"mask_chunk_{str(chunk_number).zfill(5)}.tif")
    if os.path.exists(detections_file_name):
        print(f"Skipping chunk {chunk_number}")
        return
    print(f"Submitting gpu task for chunk {chunk_number}")
    task_path = os.path.join(jobs_folder, f"segment_chunk_{str(chunk_number).zfill(5)}.sh")
    write_detection_task_for_slurm(chunk_number, task_path)
    submit_slurm_task_gpu(task_path)


def delete_extracted_chunk_by_number(number):
    import time
    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    detections_file_name = os.path.join(segmentation_folder, f"mask_chunk_{str(number).zfill(5)}.tif")
    while not os.path.exists(detections_file_name):
        time.sleep(10)
    # os.remove(chunk_file)
    print('removed', chunk_file)


def extract_chunk_by_number(number):
    if source.endswith('.ims'):
        print("extracting from ims")
        extract_chunk_from_imaris_by_number(number)
    else:
        print("extracting from tiff")
        extract_chunk_from_tiff_series_by_number(number)


def extract_detect_delete(number):
    print("Chunk number", number)
    try:
        extract_chunk_by_number(number)
    except:  # if impossible to extract, create an empty napari-compatible DF
        print(f"EXCEPTION: unable to extract chunk # {number}")
        # TODO save empty mask?
        return
    print("extracted")
    segment_one_chunk(number)
    print("sent segmentation job")
    # delete_extracted_chunk_by_number(number)
    # print("Deleted")


input_dir = sys.argv[1]  # TODO: this can be read directly from the JSON file
output_folder_sequence = sys.argv[2]
chunks_folder = sys.argv[3]
resolution_level = int(sys.argv[4])
signal_channel = int(sys.argv[5])
chunk_number = int(sys.argv[6])
username = sys.argv[7]
model = sys.argv[8]
priority = sys.argv[9]

metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
source = metadata['source']

jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")

# chunks_folder = os.path.join(output_folder_sequence, "chunks")
segmentation_folder = os.path.join(output_folder_sequence, f"model_{os.path.basename(model)}", "segmentation")
if not os.path.exists(segmentation_folder):
    os.makedirs(segmentation_folder)

extract_detect_delete(int(chunk_number))
