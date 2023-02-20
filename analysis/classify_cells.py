from glob import glob
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from analysis import settings
from analysis.find_cells import launch_cell_finder
from analysis.utils import read_info_file, update_info_file, get_resolution_level_better_than_10um

log = logging.getLogger(__name__)


def classify_cells(options):
    """
    Run cell classification via cellfinder.

    Looks for model_to_use.txt file in the root of analysis directory.
    The model_to_use.txt file contains path to the model.h5 file, relevant for this experiment.
    Model should be applicable for all brains in the experiment.
    Models will be different between experiments/projects.
    If no model path has been found, default pretrained NN (provided by brainglobe) is used.

    :param options: dict
    :return: None
    """
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas

    path = Path(options["out_name"])
    analysis_folder = path.parent.absolute()
    path_to_model_file = os.path.join(analysis_folder, settings.MODEL_PATH_FILE)
    if os.path.exists(path_to_model_file):
        with open(path_to_model_file, 'r') as f:
            path_to_model = f.read()
    else:
        path_to_model = None

    atlas = BrainGlobeAtlas(settings.ATLAS_NAME_FORMAT.format(options['allen_resolution']))
    cellfinder_output_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME
    )
    channels = options['channels']
    if channels == 1:  # no background
        signal_channels = [0]
    elif channels == 2:  # 1 signal channel, 1 background
        signal_channels = [int(not options['background_channel'])]
    else:  # multiple signal channels = other folder structure
        signal_channels = [x for x in range(options['channels']) if x != options['background_channel']]

    data_dirs_root = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
    )
    signal_folders = []
    for signal_channel in signal_channels:
        signal_folders.append(os.path.join(data_dirs_root, f"channel_{signal_channel}"))

    background_folder = os.path.join(data_dirs_root, f"channel_{options['background_channel']}")
    options_update = read_info_file(options['out_name'])
    options.update(options_update)

    launch_cell_finder(
            signal_folders,
            background_folder,
            cellfinder_output_folder,
            options['resolution'],
            options['orientation'],
            atlas,
            classify=True,
            path_to_model=path_to_model
    )

    # copy classification xml file to a folder corresponding to used model
    cell_classification_folder = os.path.join(cellfinder_output_folder, 'points')
    cell_classification_path = os.path.join(cell_classification_folder, 'cell_classification.xml')
    nn_name = path_to_model.split(os.path.sep)[-3] if path_to_model else 'default_nn'
    nn_folder = os.path.join(cell_classification_folder, nn_name)
    if not os.path.exists(nn_folder):
        os.makedirs(nn_folder)
    try:
        shutil.copyfile(cell_classification_path, nn_folder)
    except:
        log.error(f"Unable to copy file {cell_classification_path} to {nn_folder}")
    return True


def save_cells_imaris(options):
    """
    Convert spots detected in Imaris (csv) to cells.xml.

    Also saves npy file with all spots.
    :param options: dict
    :return: bool: success flag
    """
    from imlib.IO.cells import save_cells
    from imlib.cells.cells import Cell

    def _look_for_csv(parent_dir, ims_file_name):
        csv_files = []
        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file == ims_file_name.replace('.ims', '_Detailed.csv'):
                    csv_files.append(os.path.join(root, file))
                    break
        return csv_files

    ims_file_folder = os.path.dirname(options['ims_file_path'])
    ims_file_filename = os.path.basename(options['ims_file_path'])

    # look for csv files in the analysis dir
    log.debug(f"Looking for csv files in {options['out_name']}")
    found_csv_files = _look_for_csv(options['out_name'], ims_file_filename)

    log.debug(f"Found {len(found_csv_files)} csv files")
    if len(found_csv_files) == 0:
        # look for csv files next to ims file
        log.debug(f"Looking for csv files in {ims_file_folder}")
        found_csv_files = _look_for_csv(ims_file_folder, ims_file_filename)

        log.debug(f"Found {len(found_csv_files)} csv file(s)")
        if len(found_csv_files) == 0:
            return False

    log.debug(f"CSV file to convert: {found_csv_files}")
    csv_file_path = found_csv_files[0]
    log.debug("Reading csv file into data frame")
    df = pd.read_csv(csv_file_path, skiprows=3)

    options_update = read_info_file(options['out_name'])
    options.update(options_update)

    log.debug("Recalculating spots positions")
    points_coords = np.asarray(df.loc[:, ["Position Z", "Position Y", "Position X"]])
    all_detected_spots = points_coords.copy()
    all_detected_spots[:, 0] /= options['resolution'][0]
    all_detected_spots[:, 1] /= options['resolution'][1]
    all_detected_spots[:, 2] /= options['resolution'][2]

    log.debug("Saving npy file")
    npy_file_path = os.path.join(options['out_name'], settings.DETECTED_SPOTS_IMARIS_FILE_NAME)
    np.save(npy_file_path, all_detected_spots)

    # save cells to cells.xml file in cellfinder output - for classification
    cells = []
    for spot in all_detected_spots:
        cells.append(
            Cell(
                (spot[2], spot[1], spot[0]),
                Cell.UNKNOWN,
            )
        )

    log.debug("Trying to save cells.xml")
    points_dir = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME,
        'points'
    )
    if not os.path.exists(points_dir):
        os.makedirs(points_dir)

    cells_xml_file = os.path.join(points_dir, 'cells.xml')
    if os.path.exists(cells_xml_file):
        log.error(f"File {cells_xml_file} already exists, cannot overwrite!")
        return False

    save_cells(
        cells,
        cells_xml_file,
        save_csv=False,
        artifact_keep=True,
    )
    log.debug("Saved cells.xml")
    return True
