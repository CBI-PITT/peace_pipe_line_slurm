"""
Improvements TODO:
more robust bg subraction (possibly in coronal plane)
check NMI and overlap computation
merge dask branch into master, but make separation to run it only on pollux
separate branches for each machine? or local settings files?
"""
import json
import logging
import os
import sys
from pathlib import Path

from imaris_ims_file_reader import ims

from analysis import settings, local_settings
from analysis.analyze_cells import analyze_cells_cellfinder, analyze_cells_imaris, save_to_db
from analysis.classify_cells import classify_cells, save_cells_imaris
from analysis.classify_cells_fastai import classify_cells_fastai
from analysis.extract_tiff import extract_tiff_series
from analysis.fft_dask_cluster import fft_cluster
from analysis.find_cells import detect_cells
from analysis.find_cells_nn import detect_cells_nn
from analysis.guess_background_channel import guess_background
from analysis.guess_brain_orientation import guess_orientation
from analysis.pre_process import pre_process
from analysis.visualize import visualize_cells
from analysis.register_brains import register_brain, get_best_registration
from analysis.remove_stripes import calculate_first_harmonic_from_stitching
from analysis.utils import create_info_file, read_info_file, update_info_file, update_in_progress_files_json
from analysis.utils import get_resolution_level_better_than_10um, find_appropriate_atlas


log = logging.getLogger(__name__)


def do_analysis(ims_file_path, analysis_dir_this_brain, operations_to_perform):
    """
    Analysis wrapper function. Insert functions with different analysis steps here.

    Currently does registration.
    :param ims_file_path: absolute path to .ims file to be analyzed
    :param output_folder: folder to save analysis results
    :return: None
    """
    host = os.uname().nodename
    try:
        ims_file = ims(ims_file_path)
    except:  # couldn't read a file
        return

    if not os.path.exists(os.path.join(analysis_dir_this_brain, settings.INFO_FILE_NAME)):
        options = initialize_info_file(ims_file, analysis_dir_this_brain)
    else:
        options = read_info_file(analysis_dir_this_brain)  # if processing has been attempted before
        if options['ims_file_path'] != ims_file_path:  # directory has been renamed
            options['ims_file_path'] = ims_file_path
            update_info_file(analysis_dir_this_brain, options)
        # TODO: check and update processed json file

    settings_file = os.path.join(str(Path(options['out_name']).parent), 'settings.json')
    local_settings.update_settings(settings, settings_file)

    local_settings_dict = {}
    with open(settings_file, "r") as f:
        local_settings_str = f.read()
        local_settings_dict = json.loads(local_settings_str)
    operations_to_perform_adjusted = []
    for operation in operations_to_perform:
        if operation in local_settings_dict:
            operations_list = local_settings_dict[operation]
            operations_to_perform_adjusted.extend(operations_list)
        else:
            operations_to_perform_adjusted.append(operation)

    print("Actions:", operations_to_perform_adjusted)

    for operation in operations_to_perform_adjusted:
        log.info(f"Performing operation: {operation}")
        update_in_progress_files_json(host, ims_file_path, operation)  # Add operation in progress
        operation_func = getattr(sys.modules[__name__], operation)
        success = operation_func(options)
        update_in_progress_files_json(host, remove=True)  # Remove operation in progress
        if not success:
            continue  # do not update file with processed ims
        log.info(f"Successfully performed operation: {operation}. Updating processing summary file...")

        # Update file with processed brains
        with open(settings.PROCESSED_IMS_LOCATION, 'r') as f:
            input_data_str = f.read()
        processed_ims_files = json.loads(input_data_str)
        if ims_file_path in processed_ims_files:
            # update list of operations done for this file
            ops = processed_ims_files[ims_file_path]
            ops.append(operation)
            processed_ims_files[ims_file_path] = ops
        else:
            # create list of operations done for this file
            processed_ims_files[ims_file_path] = [operation]
        with open(settings.PROCESSED_IMS_LOCATION, 'w') as f:
            f.write(json.dumps(processed_ims_files))
        log.info(f"Updated processing summary file for {ims_file_path}")


def initialize_info_file(ims_file, analysis_dir_this_brain):
    orientation = guess_orientation(ims_file, save_100um_volume=True, out_dir=analysis_dir_this_brain)
    background_channel, background_channel_nmi = guess_background(
        ims_file,
        save_100um_volume=True,
        out_dir=analysis_dir_this_brain
    )
    opimal_atlas = find_appropriate_atlas(ims_file)
    atlas_resolution_to_use = max(settings.ALLEN_RESOLUTION, opimal_atlas)
    resolution_level, resolution, shape = get_resolution_level_better_than_10um(ims_file)
    options = {
        "out_name": analysis_dir_this_brain,
        "ims_file_path": ims_file.filePathComplete,
        "allen_resolution": atlas_resolution_to_use,
        "orientation": orientation,
        "channels": ims_file.Channels,
        "background_channel": background_channel,
        "background_channel_nmi": background_channel_nmi,
        "volume_100um_location": os.path.join(
            analysis_dir_this_brain,
            ims_file.fileName + settings.SUFFIX_100UM_VOLUME
        ),
        "resolution": resolution,
        "shape": shape,
        "resolution_level": resolution_level,
        "full_resolution": ims_file.resolution,
        "full_shape": ims_file.shape
    }
    create_info_file(analysis_dir_this_brain, options)
    log.info("Calculating first harmonic from stitching data")
    first_harmonic = calculate_first_harmonic_from_stitching(ims_file, analysis_dir_this_brain)
    log.info(f"First harmonic = {first_harmonic}")

    options.update({
        "first_harmonic": first_harmonic,
        # "first_harmonic_argmax": dict(first_harmonic_argmax)
    })
    update_info_file(analysis_dir_this_brain, options)
    return options
