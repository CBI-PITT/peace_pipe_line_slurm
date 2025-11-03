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
operation_folder = this_script.parent
operations_folder = operation_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings
from utils.slurm import split_slurm_array, submit_slurm_array

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def submit_stripes_removal_job_array():
    z_layers = metadata['shape'][-3]
    path_to_task = os.path.join(jobs_folder, f"remove_stripes_fft_rl{resolution_level}_c{channel}.sh")
    main_script = os.path.abspath(__file__)
    slurm_script = os.path.join(os.path.dirname(main_script), "do_stripes_removal.py")
    with open(path_to_task, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n')
        f.write(f"#SBATCH -J {username}-remove-stripes")
        f.write('\n')
        f.write(f"#SBATCH -o {jobs_folder}/slurm_%j.out")
        f.write('\n')
        f.write('\n')
        f.write("source /h20/home/lab/miniconda3/bin/activate peace")
        f.write('\n')
        f.write(f'python {slurm_script}')
        f.write(' ')
        f.write(input_dir if ' ' not in input_dir else f'"{input_dir}"')
        f.write(' ')
        f.write(save_folder if ' ' not in save_folder else f'"{save_folder}"')
        f.write(' ')
        f.write(str(resolution_level))
        f.write(' ')
        f.write(str(channel))
        f.write(' ')
        f.write('$SLURM_ARRAY_TASK_ID')
        f.write(' ')
        f.write(str(stripe_direction))
        f.write(' ')
        f.write(str(first_harmonic))
        f.write('\n')

    already_done = glob(os.path.join(save_folder, "*.tif"))
    if len(already_done):
        print("Partially processed")
        print("Processed", len(already_done), "of", z_layers)
        files = os.listdir(save_folder)
        pattern = "_z(\d+)\.tif"
        numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
        numbers = set(map(int, numbers))
        job_ids = split_slurm_array(
            path_to_task,
            z_layers,
            numbers,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=32,
            priority=priority,
        )
    else:
        job_ids = submit_slurm_array(
            path_to_task,
            z_layers,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=32,
            priority=priority,
        )
    return job_ids


def calculate_first_harmonic_one_img(image, stripes_direction='v'):
    """
    Calculate as argmax().

    stripes_period = int(np.floor(1./first_harmonic*len(quant_5)))
    """
    if stripes_direction == "h":
        axis = 1
    elif stripes_direction == "v":
        axis = 0
    else:
        raise NotImplemented("Can only automatically remove vertical or horizontal stripes")
    quant_5 = np.quantile(image.astype(np.float32), 0.05, axis=axis)
    r_fft_transf = np.fft.rfft(quant_5)/quant_5.shape[0]
    first_harmonic = np.argmax(abs(r_fft_transf)[5:]) + 5
    print(f"Calculated first harmonic: {first_harmonic}")
    return first_harmonic


def calculate_first_harmonic_from_stitching(ims_file, composites_dir):
    """
    Calculate from stitching info (shiftValues.csv, stitchData.csv). Only works for RSCM (obviously).
    """
    composites_dir = Path(composites_dir)
    shift_values_csv = str(composites_dir / 'shiftValues.csv')
    overlap_csv = str(composites_dir / 'stitchData.csv')
    if not os.path.exists(shift_values_csv) or not os.path.exists(overlap_csv):
        print("Stitching csv files do not exist")
        return
    shift_df = pd.read_csv(shift_values_csv)
    x_shift = shift_df.iloc[0]['xShift']
    overlap_df = pd.read_csv(str(overlap_csv))
    overlap = overlap_df.iloc[0]['overlap']
    stripes_width_full_resolution = 1024 - overlap + x_shift
    stripes_width_resolution_x = stripes_width_full_resolution * metadata['shape'][-1] / ims_file.shape[-1]
    calculated_first_harmonic = int(np.floor(metadata['shape'][-1] / stripes_width_resolution_x))
    return calculated_first_harmonic


def calculate_first_harmonic_from_majority():
    """
    Compute first harmonic of the striped artifact from the image.

    Computes first harmonics for each z layer (2D slice), then returns the most
    frequent value among them as the true first harmonic.
    :param analysis_dir_this_brain: str
    :return: int
    """
    import tifffile
    print("Computing 1st harmonic from the data")
    harmonics = defaultdict(int)
    imgs = sorted(glob(os.path.join(input_dir, '*.tif')))
    for z, img_name in enumerate(imgs):
        img = tifffile.imread(img_name)
        first_harmonic = calculate_first_harmonic_one_img(img, stripes_direction=stripe_direction)
        print(f"z layer {z}: first harmonic {first_harmonic}")
        harmonics[first_harmonic] += 1

    harmonics = dict(harmonics)
    most_frequent = max(harmonics, key=harmonics.get)
    return int(most_frequent)


input_dir = sys.argv[1]
save_folder = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
username = sys.argv[5]
priority = sys.argv[6]
stripe_direction = sys.argv[7]
composites_dir = sys.argv[8]

metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
output_folder_sequence = Path(save_folder).parent
jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")

first_harmonic = None

if composites_dir:  # only for RSCM
    base_input_dir = metadata['base_input_dir']
    if not base_input_dir:
        base_input_dir = input_dir
    base_metadata = json.load(open(os.path.join(base_input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
    ims_file_path = base_metadata['source']
    if ims_file_path.endswith('.ims'):
        from imaris_ims_file_reader import ims
        ims_file = ims(ims_file_path)
        first_harmonic = calculate_first_harmonic_from_stitching(ims_file, composites_dir)

if not first_harmonic:
    first_harmonic = calculate_first_harmonic_from_majority()

print("first harmonic:", first_harmonic)

submit_stripes_removal_job_array()

finished_planes = len(glob(os.path.join(save_folder, '*.tif')))
z_layers = metadata['shape'][-3]
while finished_planes < z_layers:
    print("finished", finished_planes, "of", z_layers)
    time.sleep(60)
    finished_planes = len(glob(os.path.join(save_folder, '*.tif')))
