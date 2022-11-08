from datetime import datetime
from glob import glob
import json
import logging
import os
import re
import subprocess

import numpy as np
import pandas as pd
import tifffile

import bg_space as bgs
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from cellfinder.analyse.analyse import transform_points_to_downsampled_space
from cellfinder.main import get_downsampled_space
from imlib.IO.cells import get_cells

from imaris_ims_file_reader import ims

from analysis import settings
from analysis.classify_cells import save_cells_imaris
from analysis.register_brains import register_brain, get_path_to_best_registration
from analysis.utils import get_resolution_level_better_than_10um, get_signal_channels

log = logging.getLogger(__name__)


def main(input_file_path, output_file_path, *args):
    """
    Cells are already counted in Imaris. Register to the atlas and save detailed cell info to csv (optional).

    :param input_file_path: path to a file with a list of ims files and corresponding parameters for cell counting.
    format of the input file (JSON compatible):
    [
    {
        "filename": "path/to/file1.ims",
        "spots_csv_file": "path/to/file1.csv",
        "orientation": "sal",
        "background_channel": 0,
        "allen_resolution": 10,
        "save_df": true
    },
    {
        "filename": "path/to/file2.ims",
        "spots_csv_file": "path/to/file2.csv",
        "orientation": "sal",
        "background_channel": 0,
        "allen_resolution": 10,
        "save_df": true
    },
    ...
    ]
    :param output_file_path: path to the output csv with detailed cell info
    :param args: whatever
    :return: None
    """
    with open(input_file_path, 'r') as f:
        input_data_str = f.read()

    input_data_str = input_data_str.replace("'", '"')
    input_data = json.loads(input_data_str)
    for data_entry in input_data:
        data_entry['out_filename'] = output_file_path
        analyze_cells(data_entry)


def transform_points_downsampled_to_atlas_space(
    downsampled_points, atlas, deformation_field_paths, output_filename=None
):
    field_scales = [int(1000 / resolution) for resolution in atlas.resolution]
    points = [[], [], []]
    for axis, deformation_field_path in enumerate(deformation_field_paths):
        deformation_field = tifffile.imread(deformation_field_path)
        for point in downsampled_points:
            try:
                point = [int(round(p)) for p in point]
                points[axis].append(
                    int(
                        round(
                            field_scales[axis]
                            * deformation_field[point[0], point[1], point[2]]
                        )
                    )
                )
            except IndexError:
                log.warning(
                    f'IndexError when transforming point ({point[0]},{point[1]},{point[2]}) from downsampled to atlas space.'
                )
    transformed_points = np.array(points).T

    if output_filename is not None:
        df = pd.DataFrame(transformed_points)
        df.to_hdf(output_filename, key="df", mode="w")

    return transformed_points


def analyze_cells_imaris(options):
    """
    Create csv (DataFrame) with all cells' info.

    Look for a csv file that contains ims file name in the working directory.
    If it exists, extract points from it. Transform their positions from microns to voxels.
    Save points as .npy file.

    Currently assumes that there is only 1 signal channel, for which spots were counted in Imaris.

    :param options:
    :return:
    """
    register_brain(options)
    save_cells_imaris(options)
    npy_file_path = os.path.join(options['out_name'], settings.DETECTED_SPOTS_IMARIS_FILE_NAME)
    cellfinder_output_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME
    )
    if os.path.exists(os.path.join(cellfinder_output_folder, 'points', 'cell_classification.xml')):
        transformed_filename = settings.TRANSFORMED_CELLS_FILE_NAME
    else:
        transformed_filename = settings.TRANSFORMED_SPOTS_FILE_NAME
    analyze_cells(options, npy_file_path, transformed_filename)
    return True


def analyze_cells_cellfinder(options):
    """
    Create csv (DataFrame) with all cells' info.

    Look for xml file in the cellfinder output folder.
    Take into account possible multiple channels.
    Save cells from the xml file(s) as .npy file(s).

    :param options:
    :return:
    """
    register_brain(options)

    cellfinder_output_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME
    )
    signal_channels = get_signal_channels(options['channels'], options['background_channel'])

    if len(signal_channels) == 1:
        cellfinder_output_folders = [cellfinder_output_folder]
    else:
        cellfinder_output_folders = []
        for signal_channel in range(len(signal_channels)):
            cellfinder_output_folders.append(os.path.join(cellfinder_output_folder, f"channel_{signal_channel}"))

    for cellfinder_output_folder in cellfinder_output_folders:
        cells_xml = os.path.join(cellfinder_output_folder, 'points', 'cell_classification.xml')
        if os.path.exists(cells_xml):
            out_filename = settings.CLASSIFIED_CELLS_FILE_NAME
            cells_only = True
            transformed_filename = settings.TRANSFORMED_CELLS_FILE_NAME
        else:
            cells_xml = os.path.join(cellfinder_output_folder, 'points', 'cells.xml')
            if os.path.exists(cells_xml):
                out_filename = settings.DETECTED_SPOTS_FILE_NAME
                cells_only = False
                transformed_filename = settings.TRANSFORMED_SPOTS_FILE_NAME
            else:
                return False

        npy_file_path = os.path.join(cellfinder_output_folder, out_filename)
        if not os.path.exists(npy_file_path):
            # get all detected spots into one array
            all_detections = get_cells(cells_xml, cells_only=cells_only)
            all_detected_spots = []

            for cell in all_detections:
                all_detected_spots.append([cell.z, cell.y, cell.x])

            all_detected_spots = np.asarray(all_detected_spots)
            np.save(npy_file_path, all_detected_spots)

        analyze_cells(options, npy_file_path, transformed_filename)
    return True


def analyze_cells(options, cells_npy_file_path, transformed_npy_filename):
    """
    Create csv (DataFrame) with all cells' info.

    First, try to perform registration with the highest atlas resolution
    possible for given image size (taking into account NiftyReg limitations).
    If registered atlas exists for each channel, skips the registration.
    Among all registration outputs, take the best one (by NMI).
    Get points either from cellfinder output (xml) or from Imaris spot counting (csv), create np.array from them.
    Transform points to atlas space.
    Create DataFrame.

    :param options:
    {
        "out_name": str,
        "ims_file_path": str,
        "allen_resolution": int,
        "orientation": str,
        "channels": int,
        "background_channel": int,
        "background_channel_nmi": int,
        "volume_100um_location": str
    }
    :return: csv file path
    """
    cellfinder_output_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME
    )
    info_file_path = os.path.join(options['out_name'], settings.INFO_FILE_NAME)

    # transform points to atlas space
    log.info('Creating DataFrame for detected cells...')
    with open(info_file_path, "r") as f:
        options_str = f.read()
    options = json.loads(options_str)

    source_space = bgs.AnatomicalSpace(
        options['orientation'],
        shape=options['shape'],
        resolution=options['resolution'],
    )

    # best_registration_dir_prefix = get_path_to_best_registration(options['out_name'])
    registration_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME,
        f"registration"
        # f"registration_{best_registration_dir_prefix}"
    )

    deformation_field_paths = [
        os.path.join(registration_folder, 'deformation_field_0.tiff'),
        os.path.join(registration_folder, 'deformation_field_1.tiff'),
        os.path.join(registration_folder, 'deformation_field_2.tiff')
    ]

    atlas = BrainGlobeAtlas('allen_mouse_{}um'.format(options['allen_resolution']))

    downsampled_space = get_downsampled_space(
        atlas,
        os.path.join(registration_folder, 'boundaries.tiff')
    )

    all_detected_spots = np.load(cells_npy_file_path)
    all_detected_spots_downsampled = transform_points_to_downsampled_space(
        all_detected_spots, downsampled_space, source_space
    )
    all_detected_spots_transformed = transform_points_downsampled_to_atlas_space(
        all_detected_spots_downsampled, atlas, deformation_field_paths
    )
    np.save(
        os.path.join(cellfinder_output_folder, transformed_npy_filename),
        all_detected_spots_transformed
    )

    # For each point, get atlas label
    label_ids = []
    empty_points = []
    error_points = []
    good_points = []
    df_data = []

    for ind in range(all_detected_spots_transformed.shape[0]):
        label_id = 0
        structure_code = ''
        structure_name = ''

        if np.any(all_detected_spots_transformed[ind] < 0):
            error_points.append([
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ])
            log.warning("Point with negative coordinates")
            continue

        try:  # some points are outside atlas
            atlas_value = atlas.annotation[
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ]
        except IndexError as e:
            error_points.append([
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ])
        else:  # if we didn't get exception
            df_row = atlas.lookup_df.index[atlas.lookup_df['id'] == atlas_value]
            if not df_row.empty:  # such atlas_value exists
                row_values = atlas.lookup_df.iloc[df_row]
                structure_name = row_values['name'].values[0]
                structure_code = row_values['acronym'].values[0]
                label_id = atlas_value
                good_points.append([
                    all_detected_spots_transformed[ind, 0],
                    all_detected_spots_transformed[ind, 1],
                    all_detected_spots_transformed[ind, 2]
                ])
            else:  # no such atlas_value (it's usually 0 in this case)
                empty_points.append([
                    all_detected_spots_transformed[ind, 0],
                    all_detected_spots_transformed[ind, 1],
                    all_detected_spots_transformed[ind, 2]
                ])
        finally:  # it will run either way
            if any([x < 0 for x in all_detected_spots_transformed[ind, :]]):
                label_id = 0
                structure_code = ''
                structure_name = ''

            label_ids.append(label_id)
            data_entry = [
                ind,
                *list(all_detected_spots[ind, :] * options["resolution"]),
                options["resolution_level"],
                *list(options["resolution"]),
                *list(all_detected_spots[ind, :]),
                atlas.atlas_name,
                *list(all_detected_spots_downsampled[ind, :]),
                *list(all_detected_spots_transformed[ind, :]),
                label_id,
                structure_code,
                structure_name,
                options["ims_file_path"]
            ]
            df_data.append(data_entry)

    log.info(
        f'''
        Points within atlas dimensions, without a label: {len(empty_points) / len(label_ids) * 100} %\n
        Points outside atlas dimensions, without a label: {len(error_points) / len(label_ids) * 100} %\n
        Total points without a label: {(len(empty_points) + len(error_points)) / len(label_ids) * 100}
        '''
    )
    np.save(os.path.join(cellfinder_output_folder, f'{transformed_npy_filename}_empty.npy'), empty_points)
    np.save(os.path.join(cellfinder_output_folder, f'{transformed_npy_filename}_error.npy'), error_points)
    np.save(os.path.join(cellfinder_output_folder, f'{transformed_npy_filename}_good.npy'), good_points)

    # create a DataFrame
    df_column_names = [
        'ind',
        'z_raw_um', 'y_raw_um', 'x_raw_um',
        'resolution_level_used',
        'resolution_used_z', 'resolution_used_y', 'resolution_used_x',
        'z_raw_px', 'y_raw_px', 'x_raw_px',
        'atlas',
        'z_downsampled_px', 'y_downsampled_px', 'x_downsampled_px',
        'z_atlas_px', 'y_atlas_px', 'x_atlas_px',
        'atlas_label_id',
        'structure_acronym',
        'structure_name',
        'path_to_file'
    ]

    df = pd.DataFrame(df_data, columns=df_column_names)
    timestamp = datetime.now().strftime(settings.TIMESTAMP_FORAMT)
    output_csv_file_path = os.path.join(options['out_name'], settings.CELLS_DATAFRAME_NAME_PATTERN.format(timestamp))
    df.to_csv(output_csv_file_path)
    log.info('Created DataFrame for detected cells')


if __name__ == "__main__":
    main(
        # '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/analyze.txt',
        # '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/register_to_atlas.txt',
        '/CBI_Hive/Public/freyberg-z/02CL22/analysis/register.txt',
        ''
        # '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/analysis/cells_detailed_info.csv'
    )
