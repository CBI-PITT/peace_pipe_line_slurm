from collections import defaultdict
import json
import os
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
from utils.slurm import submit_slurm_indices
from utils.z_range import existing_z_indices, normalize_z_range

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def submit_stripes_removal_job_array():
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

    return submit_slurm_indices(
        path_to_task,
        selected_indices,
        existing_outputs=existing_z_indices(save_folder),
        partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
        cores=1,
        memory=32,
        priority=priority,
    )


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
        raise ValueError("Can only automatically remove vertical or horizontal stripes")
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
    if not imgs:
        raise FileNotFoundError(f"No TIFF files found in {input_dir}")
    for z, img_name in enumerate(imgs):
        img = tifffile.imread(img_name)
        first_harmonic = calculate_first_harmonic_one_img(img, stripes_direction=stripe_direction)
        print(f"z layer {z}: first harmonic {first_harmonic}")
        harmonics[first_harmonic] += 1

    harmonics = dict(harmonics)
    most_frequent = max(harmonics, key=harmonics.get)
    return int(most_frequent)


def record_first_harmonic(provenance_path, first_harmonic, method, scope):
    with open(provenance_path, 'r') as f:
        provenance = json.load(f)
    provenance.setdefault('process', {})['results'] = {
        'first_harmonic': first_harmonic,
        'first_harmonic_method': method,
        'first_harmonic_scope': scope,
    }
    temporary_path = f"{provenance_path}.tmp"
    with open(temporary_path, 'w') as f:
        json.dump(provenance, f)
    os.replace(temporary_path, provenance_path)


input_dir = sys.argv[1]
save_folder = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
username = sys.argv[5]
priority = sys.argv[6]
stripe_direction = sys.argv[7]
composites_dir = sys.argv[8]
provenance_path = sys.argv[9]
requested_z_start = int(sys.argv[10])
requested_z_end = int(sys.argv[11])

metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
selection = normalize_z_range(metadata, requested_z_start, requested_z_end)
selected_indices = list(range(selection['start'], selection['end']))
output_folder_sequence = Path(save_folder).parent
jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
if not selection['is_full']:
    jobs_folder += f"_z{selection['start']}-{selection['end']}"

first_harmonic = None
first_harmonic_method = None
first_harmonic_scope = None

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
        if first_harmonic:
            first_harmonic_method = 'stitching_metadata'
            first_harmonic_scope = 'stitching_metadata'

if not first_harmonic:
    first_harmonic = calculate_first_harmonic_from_majority()
    first_harmonic_method = 'image_majority'
    first_harmonic_scope = 'all_available_input_z'

print("first harmonic:", first_harmonic)
record_first_harmonic(
    provenance_path,
    first_harmonic,
    first_harmonic_method,
    first_harmonic_scope,
)

job_ids = submit_stripes_removal_job_array()
missing = set(selected_indices) - existing_z_indices(save_folder)
if missing and not job_ids:
    raise RuntimeError(
        f"No SLURM jobs were submitted for missing z indices: {sorted(missing)[:10]}"
    )

while True:
    completed = existing_z_indices(save_folder)
    missing = set(selected_indices) - completed
    if not missing:
        break
    print("finished", len(selected_indices) - len(missing), "of", len(selected_indices))
    time.sleep(60)
