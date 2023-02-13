import json
import logging
import os
import re
import subprocess

import dask
from dask import delayed
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.metrics import normalized_mutual_information

from analysis import settings
from analysis.pre_process import pre_process_brain_faster
from analysis.remove_stripes import calculate_first_harmonic_from_stitching
from analysis.background_subtraction import subtract_background_iterative
from analysis.utils import (
    ensure_tiffs_extracted,
    read_info_file,
    create_registration_info_file,
    read_registration_info_file,
    update_registration_info_file
)

log = logging.getLogger(__name__)


FIRST_HARMONIC = None


def main(input_path, output_path='', *args):
    """
    Register to the atlas.

    Atlas resolution (if not specified) is based on image dimensions
    (should all be < 2048 at 10um resolution - for NiftyReg).

    :param input_file_path: path to a txt file with a list of ims files.
    format of the input file (JSON compatible):
    [
    {
        "filename": "path/to/file1.ims",
        "orientation": "sal",
        "register_channel": 0,
        "allen_resolution": 10,
    },
    {
        "filename": "path/to/file2.ims",
        "orientation": "sal",
        "register_channel": 0,
        "allen_resolution": 10,
    },
    ...
    ]
    :param output_file_path: path to the registration folder
    if not specified, registration folder will be within the 'analysis' folder, which is next to the ims file
    :param args: whatever
    :return: None
    """
    with open(input_path, 'r') as f:
        input_data_str = f.read()

    input_data_str = input_data_str.replace("'", '"')
    input_data = json.loads(input_data_str)
    for data_entry in input_data:
        data_entry['out_name'] = output_path
        register_brain(data_entry)


def register_brain_one_channel(options, ims_file, channel=0, folder_prefix=None):
    """
    Register specified channel.

    :param options, dict
      example: options = {
        "out_name": "/path/to/analysis/ims_file_name_folder/",
        "allen_resolution": 10,
        "resolution": [10, 10, 10],
        "orientation": "sal",
    }
    :param ims_file, instance of ims class
      example: ims_file = ims("/path/to/file.ims")
    """
    log.info(f"Registering channel {channel}")
    analysis_dir_this_brain = options['out_name']
    options_update = read_info_file(options['out_name'])
    options.update(options_update)
    atlas = settings.ATLAS_NAME_FORMAT.format(options["allen_resolution"])
    orientation = options["orientation"]
    out_directory = os.path.join(analysis_dir_this_brain, 'resolution_level_x')

    if folder_prefix:
        # registering pre-processed data
        brainreg_input = os.path.join(out_directory, folder_prefix)
        brainreg_output_folder = os.path.join(out_directory, f"registration_{folder_prefix}")
        log.info(f"Registering pre-processed data at {brainreg_output_folder}")
        if not os.path.exists(os.path.join(brainreg_output_folder, 'registered_atlas.tiff')):
            launch_registration(brainreg_input, brainreg_output_folder, atlas, options["resolution"], orientation)
        else:
            log.warning(f'{brainreg_input}: Registered atlas exists. Skipping registration.')

        if os.path.exists(os.path.join(brainreg_output_folder, 'registered_atlas.tiff')):
            overlap, nmi = compute_overlap_and_nmi_with_atlas(brainreg_output_folder)
            return overlap, nmi
        return None, None

    # registering original (not pre-processed) data
    brainreg_output_folder = os.path.join(out_directory, f'registration_channel_{channel}')
    brainreg_input = os.path.join(out_directory, f"channel_{channel}")

    # Extract tiff stack or series
    ensure_tiffs_extracted(channel, options)

    if not os.path.exists(os.path.join(brainreg_output_folder, 'registered_atlas.tiff')):
        launch_registration(brainreg_input, brainreg_output_folder, atlas, options["resolution"], orientation)
    else:
        log.info(f'{brainreg_input}: Registered atlas exists. Skipping registration.')

    if os.path.exists(os.path.join(brainreg_output_folder, 'registered_atlas.tiff')):
        overlap, nmi = compute_overlap_and_nmi_with_atlas(brainreg_output_folder)
        return overlap, nmi

    return None, None


def register_brain(options, folder_prefix=None):
    """
    Register all channels.
    """
    from imaris_ims_file_reader import ims

    log.info(f"Registering all channels at {options['out_name']}")
    ims_file_path = options["ims_file_path"]
    ims_file = ims(ims_file_path)

    info_file_path = os.path.join(options['out_name'], settings.INFO_FILE_NAME)

    if os.path.exists(info_file_path):
        old_options = read_info_file(options['out_name'])
        options.update(old_options)
    else:
        from analysis.main import initialize_info_file
        initialize_info_file(ims_file, options['out_name'])

    registration_info_file_path = os.path.join(options['out_name'], settings.REGISTRATION_INFO_FILE_NAME)
    if not os.path.exists(registration_info_file_path):
        create_registration_info_file(options['out_name'])
        registration_info = {}
    else:
        registration_info = read_registration_info_file(options['out_name'])
    for channel in range(ims_file.Channels):
        registration_info_this_channel = registration_info.get(f'channel_{channel}')
        if not registration_info_this_channel:
            overlap, nmi = register_brain_one_channel(options, ims_file, channel=channel, folder_prefix=folder_prefix)
            if overlap is None or nmi is None:
                log.error(f"Overlap or NMI wasn't computed for channel {channel}")
                continue  # TODO: or return False here and not continue with other channels ???
            registration_info[f'channel_{channel}'] = {'overlap': overlap, 'nmi': nmi}
            log.info(f"Updating registration info for channel {channel}")
            update_registration_info_file(options['out_name'], registration_info)
        else:
            log.warning(f"Registration info for channel {channel} exists. Skipping.")
    return True  # TODO: find more intelligent way (flags) to catch errors


def get_best_registration(options):
    """
    Attempt different pre-processings to improve the registration.
    """
    from imaris_ims_file_reader import ims

    ims_file = ims(options["ims_file_path"])
    analysis_dir_this_brain = options['out_name']
    registration_info = read_registration_info_file(analysis_dir_this_brain)
    use_dask = os.uname().nodename in settings.DASK_ALLOWED_NODES

    from analysis.pre_process import retrieve_fft_first_harmonic
    retrieve_fft_first_harmonic(options)

    for channel in range(ims_file.Channels):
        ensure_tiffs_extracted(channel, options)
        pre_processed_data_dirs = pre_process_brain_faster(
            os.path.join(analysis_dir_this_brain, "resolution_level_x"),
            channel=channel,
            use_dask=use_dask
            # denoise=settings.DENOISE_FFT
        )
        for data_dir in pre_processed_data_dirs:
            registration_info_this_dir = registration_info.get(data_dir)
            if not registration_info_this_dir:
                overlap, nmi = register_brain_one_channel(options, ims_file, channel=0, folder_prefix=data_dir)
                if overlap is None or nmi is None:
                    log.error(f"Overlap or NMI wasn't computed for dir {data_dir}")
                    continue
                registration_info[data_dir] = {'overlap': overlap, 'nmi': nmi}
                update_registration_info_file(analysis_dir_this_brain, registration_info)
            else:
                log.warning(f"Registration info exists for the dir {data_dir}. Skipping.")


def launch_registration(input_folder, output_folder, atlas, resolution, orientation, run_in_background=False):
    #  Run brainreg
    log.info('Starting registration CL tool ...')

    cmd = [
        'brainreg', input_folder, output_folder,
        '-v', str(resolution[0]), str(resolution[1]), str(resolution[2]),
        '--orientation', orientation,
        '--atlas', atlas,
        '--save-original-orientation'
    ]

    if run_in_background:
        subprocess.Popen(cmd)
        return

    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        if ret.stdout:
            log.error(ret.stdout.decode())
        log.error('Registration failed')
        return

    log.info('Finished registration')


def compute_overlap_and_nmi_with_atlas(registration_folder):
    """
    For test brain:
        - overlap ~= 41%
        - NMI ~= 1.1975
    """
    log.debug(f"Computing overlap and NMI for {registration_folder}")
    log.debug(f"Reading atlas image")
    try:
        with open(os.path.join(registration_folder, 'brainreg.json')) as f:
            txt = f.read()
    except FileNotFoundError:
        with open(os.path.join(registration_folder, 'cellfinder.json')) as f:
            txt = f.read()
    registration_params = json.loads(txt)
    atlas_name = registration_params['atlas']
    allen_resolution = re.findall(r'\d+', atlas_name)[0]
    atlas_ref = tifffile.imread(settings.ATLAS_REFERENCE_PATH_FORMAT.format(allen_resolution))
    log.debug(f"Reading atlas image - done")
    log.debug(f"Reading downsampled_standard image")
    downsampled_standard = tifffile.imread(os.path.join(registration_folder, "downsampled_standard.tiff"))
    log.debug(f"Reading downsampled_standard image - done")
    downsampled_standard_wo_bckgnd_path = os.path.join(registration_folder, "downsampled_standard_wo_background.tiff")
    if os.path.exists(downsampled_standard_wo_bckgnd_path):
        downsampled_standard_wo_bckgnd = tifffile.imread(downsampled_standard_wo_bckgnd_path)
    else:
        log.debug("Subtracting background from downsampled_standard")
        img_pixels = downsampled_standard.shape[0] * downsampled_standard.shape[1]

        def my_ceil(a, precision=0):
            return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)

        def process_layer(ind, layer):
            print(f"processing layer {ind}")
            layer_copy = layer.astype(np.float32)
            percent0 = (layer_copy[layer_copy == 0].shape[0]) / img_pixels
            q = np.quantile(layer_copy, my_ceil(percent0, 2))
            layer_copy[layer_copy == 0] = q
            layer_copy = gaussian_filter(layer_copy, sigma=5)
            mask = subtract_background_iterative(layer_copy, min_percent_zeros=np.ceil(percent0 * 1.5 * 100), mask_only=True)
            layer[mask == 0] = 0
            return layer

        layers = list(downsampled_standard)
        processed_layers = [delayed(process_layer)(i, l) for i, l in enumerate(layers)]
        processed_layers = dask.compute(processed_layers)
        downsampled_standard_wo_bckgnd = np.squeeze(np.asarray(processed_layers))

        log.debug(f"Writing downsampled_standard image w/o background")
        tifffile.imwrite(downsampled_standard_wo_bckgnd_path, downsampled_standard_wo_bckgnd)
        log.debug(f"Writing downsampled_standard image w/o background - done")
    log.debug(f"Computing overlap")
    overlap = downsampled_standard_wo_bckgnd[np.logical_and(downsampled_standard_wo_bckgnd != 0, atlas_ref != 0)]
    pixels = atlas_ref.shape[0] * atlas_ref.shape[1] * atlas_ref.shape[2]
    overlap_percent = np.round((overlap.shape[0] / pixels * 100), decimals=2)
    log.debug(f"Computing overlap - done")
    log.debug(f"Computing nmi")
    nmi = normalized_mutual_information(downsampled_standard_wo_bckgnd, atlas_ref)
    log.debug(f"Computing nmi - done")
    log.debug(f"Computed overlap and NMI for {registration_folder}")
    return float(overlap_percent), float(nmi)


def get_path_to_best_registration(analysis_dir_this_brain):
    log.debug(f"Getting best registration in {analysis_dir_this_brain}")
    dataset_info = read_info_file(analysis_dir_this_brain)
    registration = dataset_info['registration']
    max_nmi = max(registration.values(), key=lambda x: x['nmi'])
    data_dir = [x for x in registration.keys() if registration[x]['nmi'] == max_nmi]
    log.debug(f"Best registration is {data_dir[0]}")
    return data_dir[0]


if __name__ == "__main__":
    main(
        # '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/analyze.txt',
        # '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/register_to_atlas.txt',
        '/CBI_Hive/Acquire/Klimstra/03CL02/analysis/register_brains.txt',
        '/CBI_Hive/Acquire/Klimstra/03CL02/'
        # '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/analysis/cells_detailed_info.csv'
    )
