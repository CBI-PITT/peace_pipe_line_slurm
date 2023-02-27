from datetime import datetime
from glob import glob
import json
import logging
import os
import re
import subprocess
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from analysis import settings, local_settings
from analysis.classify_cells import save_cells_imaris
from analysis.register_brains import register_brain, get_path_to_best_registration
from analysis.utils import (
    get_resolution_level_better_than_10um,
    get_signal_channels,
)
from analysis.db_utils import (
    get_db_location_sqlalchemy, get_db_connection, create_metadata_table, create_metadata_record, create_cell_table,
    metadata_table_exists, metadata_record_exists, Metadata
)

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
    analyze_cells(options)
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
    from imlib.IO.cells import get_cells

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

        analyze_cells(options)
    return True


def analyze_cells(options):
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
    from cellfinder.analyse.analyse import transform_points_to_downsampled_space
    from cellfinder.main import get_downsampled_space
    import bg_space as bgs
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas
    import ulid

    settings_file = os.path.join(str(Path(options['out_name']).parent), 'settings.json')
    local_settings.update_settings(settings, settings_file)
    cellfinder_output_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME
    )
    info_file_path = os.path.join(options['out_name'], settings.INFO_FILE_NAME)
    ims_file_path = options['ims_file_path']
    path_to_classification_df = os.path.join(cellfinder_output_folder, 'points', f'predictions_{settings.PYTORCH_MODEL_NAME}_{settings.PYTORCH_MODEL_VERSION}.csv')

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

    classification_df = pd.read_csv(path_to_classification_df)
    classification_df_coords = classification_df[["axis-0", "axis-1", "axis-2"]]
    all_detected_spots = classification_df_coords.to_numpy()
    all_detected_spots_downsampled = transform_points_to_downsampled_space(
        all_detected_spots, downsampled_space, source_space
    )
    all_detected_spots_transformed = transform_points_downsampled_to_atlas_space(
        all_detected_spots_downsampled, atlas, deformation_field_paths
    )
    np.save(
        os.path.join(cellfinder_output_folder, "all_detected_spots_transformed.npy"),
        all_detected_spots_transformed
    )
    all_detected_spots_downsampled = np.round(all_detected_spots_downsampled).astype(int)
    # For each point, get atlas label
    label_ids = []
    empty_points = []
    error_points = []
    good_points = []
    df_data = []

    con = sqlite3.connect(settings.DB_LOCATION)
    cur = con.cursor()
    metadata_record_id = cur.execute(f'SELECT id FROM metadata WHERE file_path="{ims_file_path}"').fetchone()  # TODO: use LIKE
    con.close()
    if len(metadata_record_id):
        metadata_record_id = metadata_record_id[0]
    signal_channels = get_signal_channels(options["channels"], options["background_channel"])
    signal_channel = signal_channels[0]  # TODO handle multiple signal channels

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
                ulid.new(),  # uuid  # TODO: create when saving imaris points?
                0,  # time_point
                signal_channel,  # channel
                all_detected_spots[ind, 0] * options['resolution'][0],  # 'z_raw'  # TODO take from original imaris points?
                all_detected_spots[ind, 1] * options['resolution'][1],  # 'y_raw',
                all_detected_spots[ind, 2] * options['resolution'][2],  # 'x_raw',
                'um',  # 'raw_coord_units'
                int(round(all_detected_spots[ind, 0] * options['resolution'][0] / options['full_resolution'][0])),  # 'z_raw_px'  # TODO take from original imaris points?
                int(round(all_detected_spots[ind, 1] * options['resolution'][1] / options['full_resolution'][1])),  # 'y_raw_px'
                int(round(all_detected_spots[ind, 2] * options['resolution'][2] / options['full_resolution'][2])),  # 'x_raw_px'
                int(classification_df.loc[ind, "nn_decoded"] == 'cell') if classification_df is not None else 0, # 'is_cell',
                '',  # 'type',
                atlas.atlas_name,  # 'atlas_name',
                options["allen_resolution"],  # 'atlas_resolution',
                all_detected_spots_downsampled[ind, 0],  # 'z_downsampled',
                all_detected_spots_downsampled[ind, 1],  # 'y_downsampled',
                all_detected_spots_downsampled[ind, 2],  # 'x_downsampled',
                all_detected_spots_transformed[ind, 0] * atlas.resolution[0],  # 'z_transformed',
                all_detected_spots_transformed[ind, 1] * atlas.resolution[1],  # 'y_transformed',
                all_detected_spots_transformed[ind, 2] * atlas.resolution[2],  # 'x_transformed',
                'um',  # transformed_coord_units
                int(round(all_detected_spots_transformed[ind, 0])),  # 'z_transformed_px',
                int(round(all_detected_spots_transformed[ind, 1])),  # 'y_transformed_px',
                int(round(all_detected_spots_transformed[ind, 2])),  # 'x_transformed_px',
                structure_name,  # 'atlas_structure_name',
                structure_code,  # 'atlas_structure_acronym',
                label_id,  # 'atlas_structure_number',
                metadata_record_id,  # 'metadata'
            ]
            df_data.append(data_entry)

    log.info(
        f'''
        Points within atlas dimensions, without a label: {len(empty_points) / len(label_ids) * 100} %\n
        Points outside atlas dimensions, without a label: {len(error_points) / len(label_ids) * 100} %\n
        Total points without a label: {(len(empty_points) + len(error_points)) / len(label_ids) * 100}
        '''
    )
    np.save(os.path.join(cellfinder_output_folder, f'all_detected_spots_transformed_empty.npy'), empty_points)
    np.save(os.path.join(cellfinder_output_folder, f'all_detected_spots_transformed_error.npy'), error_points)
    np.save(os.path.join(cellfinder_output_folder, f'all_detected_spots_transformed_good.npy'), good_points)

    # create a DataFrame
    df_column_names = [
        'uuid',
        'time_point',
        'channel',
        'z_raw',
        'y_raw',
        'x_raw',
        'raw_coord_units',
        'z_raw_px',
        'y_raw_px',
        'x_raw_px',
        'is_cell',
        'type',
        'atlas_name',
        'atlas_resolution',
        'z_downsampled',
        'y_downsampled',
        'x_downsampled',
        'z_transformed',
        'y_transformed',
        'x_transformed',
        'transformed_coord_units',
        'z_transformed_px',
        'y_transformed_px',
        'x_transformed_px',
        'atlas_structure_name',
        'atlas_structure_acronym',
        'atlas_structure_number',
        'metadata'
    ]

    df = pd.DataFrame(df_data, columns=df_column_names)
    timestamp = datetime.now().strftime(settings.TIMESTAMP_FORAMT)
    output_csv_file_path = os.path.join(options['out_name'], settings.CELLS_DATAFRAME_NAME_PATTERN.format(timestamp))
    print('output_csv_file_path', output_csv_file_path)
    df.to_csv(output_csv_file_path)
    log.info('Created DataFrame for detected cells')

    df = df.astype(
        {"uuid": str, "z_raw": float, "y_raw": float, "x_raw": float, "raw_coord_units": str, "z_raw_px": int,
         "y_raw_px": int, "x_raw_px": int, "is_cell": int, "type": str, "atlas_name": str, "atlas_resolution": str,
         "z_downsampled": int, "y_downsampled": int, "x_downsampled": int, "z_transformed": float,
         "y_transformed": float, "x_transformed": float, "transformed_coord_units": str, "z_transformed_px": int,
         "y_transformed_px": int, "x_transformed_px": int, "atlas_structure_name": str, "atlas_structure_acronym": str,
         "atlas_structure_number": int, "metadata": int}
    )
    return df


def save_to_db(options):
    from sqlalchemy import create_engine

    settings_file = os.path.join(str(Path(options['out_name']).parent), 'settings.json')
    local_settings.update_settings(settings, settings_file)
    print("db_location", settings.DB_LOCATION)
    print("db_type", settings.DB_TYPE)
    if metadata_record_exists(options["ims_file_path"]):
        print("Metadata record exists")
        return False
    save_metadata_to_db(options)
    create_cell_table()
    df = analyze_cells(options)
    df = df[["uuid", "time_point", "channel", "z_raw", "y_raw", "x_raw", "raw_coord_units",
             "z_raw_px", "y_raw_px", "x_raw_px", "is_cell", "type", "atlas_name", "atlas_resolution",
             "z_downsampled", "y_downsampled", "x_downsampled", "z_transformed", "y_transformed", "x_transformed",
             "transformed_coord_units", "z_transformed_px", "y_transformed_px", "x_transformed_px",
             "atlas_structure_name", "atlas_structure_acronym", "atlas_structure_number", "metadata"]]
    location = get_db_location_sqlalchemy()
    engine = create_engine(location)
    con = engine.connect()
    df.to_sql('cell', con, if_exists='append', index=False)
    con.commit()
    con.close()
    log.info('Created database records for detected cells')
    print("Done saving to db")
    return True


def save_metadata_to_db(options):
    # TODO: create database if not exists? (for MySQL only)
    # create table metadata if not exists
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy import create_engine

    class Base(DeclarativeBase):
        pass

    location = get_db_location_sqlalchemy()
    engine = create_engine(location)

    if not metadata_table_exists():
        print("creating metadata table")
        # Base.metadata.create_all(engine)
        print("created metadata table")
        create_metadata_table()
    # insert new metadata record
    if not metadata_record_exists(options["ims_file_path"]):
        print("Creating metadata record")
        create_metadata_record(options["ims_file_path"])
        print("Created metadata record")
