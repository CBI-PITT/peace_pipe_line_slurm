import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import tifffile

from analysis.settings import (
    DEEPBLINK_CHUNK_SIZE,
    DEEPBLINK_MODEL_PATH,
    DEEPBLINK_OUT_FOLDER_NAME,
    RESOLUTION_LEVEL_FOLDER_NAME,
    DASK_ALLOWED_NODES,
)
from analysis.utils import get_signal_channels, tiff_series_to_ram, split_in_chunks_nd_no_overlap, chunks_to_tiffs, dask_chunks_to_tiffs

log = logging.getLogger(__name__)
USE_DASK = os.uname().nodename in DASK_ALLOWED_NODES


def detect_cells_nn(options):
    from imaris_ims_file_reader import ims

    tst = datetime.now()
    analysis_dir_this_brain = options["out_name"]
    out_directory = os.path.join(analysis_dir_this_brain, RESOLUTION_LEVEL_FOLDER_NAME)
    signal_channels = get_signal_channels(options["channels"], options["background_channel"])
    signal_folders = []
    for signal_channel in signal_channels:
        signal_folders.append(os.path.join(out_directory, f"channel_{signal_channel}"))

    deepblink_folder = os.path.join(out_directory, DEEPBLINK_OUT_FOLDER_NAME)
    if not os.path.exists(deepblink_folder):
        os.makedirs(deepblink_folder)

    for ind, signal_channel in enumerate(signal_channels):
        out_csv_path = os.path.join(deepblink_folder, f'channel_{signal_channel}_cells.csv')
        if os.path.exists(out_csv_path):
            log.warning(f"Output dataframe already exists at {out_csv_path}. Skipping.")
            continue
        print("Detecting cells in channel", signal_channel)
        ratios = (np.array(options['shape']) / np.array(DEEPBLINK_CHUNK_SIZE)).astype('int') + 1
        patchify_chunks_shape = (*list(ratios), *DEEPBLINK_CHUNK_SIZE)
        origin_coords = get_origin_coords(3, patchify_chunks_shape, DEEPBLINK_CHUNK_SIZE)
        chunks_folder = os.path.join(out_directory, f"channel_{signal_channel}_chunks")
        if not os.path.exists(chunks_folder):
            os.makedirs(chunks_folder)
        np.save(
            os.path.join(chunks_folder, "origin_coords.npy"),
            origin_coords
        )
        if USE_DASK:
            ims_file = ims(options['ims_file_path'], ResolutionLevelLock=options['resolution_level'])
            ims_file_dask = da.array(ims_file)
            tiffstack = ims_file_dask[0,signal_channel,:,:,:]
            chunk_indices = get_chunk_indices(origin_coords, DEEPBLINK_CHUNK_SIZE)
            dask_chunks_to_tiffs(tiffstack, chunk_indices, chunks_folder)
        else:
            tiffstack = tiff_series_to_ram(signal_folders[ind], options['shape'], np.uint16)
            if tiffstack is None:
                continue
            chunks = split_in_chunks_nd_no_overlap(tiffstack, DEEPBLINK_CHUNK_SIZE)
            chunks_to_tiffs(chunks, chunks_folder)

        detect_cells_deepblink(chunks_folder)
        convert_to_napari_format(chunks_folder)
        df = merge_df(chunks_folder, origin_coords)
        df.to_csv(out_csv_path)

    tfi = datetime.now()
    print("Total time spent on detection", tfi - tst)


def launch_deepblink(model_path, img_path):
    cmd = [
        'deepblink', 'predict', '--model', model_path, '--input', img_path
    ]
    ret = subprocess.run(cmd)


def detect_cells_deepblink(chunks_folder):
    print("Splitting stack into chunks")
    files = sorted(glob(os.path.join(chunks_folder, '*.tif')))
    total_chunks = len(files)
    print("Starting detection")
    for ind, file in enumerate(files):
        if os.path.exists(os.path.join(chunks_folder, os.path.basename(file).replace('tif', 'csv'))):
            print(f"Skipping chunk {ind}")
            continue
        print(f"Detecting cells in chunk {ind} of {total_chunks}")
        tstl = datetime.now()
        launch_deepblink(DEEPBLINK_MODEL_PATH, file)
        print("Chunk finished in", datetime.now() - tstl)


def convert_to_napari_format(chunks_folder):
    """
    Change columns in csv file to make it readable with napari.

    :param chunks_folder:
    :return:
    """
    csv_files = sorted(glob(os.path.join(chunks_folder, '*.csv')))
    for csv_file in csv_files:
        if csv_file.startswith('napari'):
            continue
        napari_csv_file_path = os.path.join(os.path.dirname(csv_file), f"napari_{os.path.basename(csv_file)}")
        if os.path.exists(napari_csv_file_path):
            continue
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError as e:
            print("Warning: ", e)
            continue
        df2 = pd.DataFrame()
        df2['index'] = list(range(df.shape[0]))
        try:
            zvals = df['z'].tolist()
            yvals = df['y [px]'].tolist()
            xvals = df['x [px]'].tolist()
        except KeyError as e:
            print(e)
            continue

        df2['axis-0'] = zvals
        df2['axis-1'] = xvals
        df2['axis-2'] = yvals
        df2.to_csv(napari_csv_file_path)


def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    """
    Get coordinates of each chunk origin.

    TODO: only 3D now, make compatible with 2D

    :param ndim:
    :param chunk_shape:
    :param patches_shape:
    :return:
    """
    coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
    coords = np.empty(coords_shape, dtype=np.uint16)
    print(" coords shape", coords.shape)
    for z in range(coords.shape[0]):
        for y in range(coords.shape[1]):
            for x in range(coords.shape[2]):
                coords[z, y, x, :] = np.array((
                    z * chunk_size[0],
                    y * chunk_size[1],
                    x * chunk_size[2]
                ))
    coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
    print("final coords shape", coords.shape)
    return coords


def merge_df(chunks_folder, origin_coords):
    """
    Transform coordinates from chunk space to raw image space.

    :return:
    """

    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']  # TODO: ndim
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(chunks_folder, 'napari*.csv')))
    print("CSV files", len(csv_files), csv_files[:3])
    for chunk_file in csv_files:
        current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
        print("Processing", current_chunk)
        chunk_df = pd.read_csv(chunk_file)
        chunk_df_corrected = pd.DataFrame()
        z_values = chunk_df[['axis-0']].to_numpy()
        y_values = chunk_df[['axis-1']].to_numpy()
        x_values = chunk_df[['axis-2']].to_numpy()
        z_values += origin_coords[current_chunk, 0]
        y_values += origin_coords[current_chunk, 1]
        x_values += origin_coords[current_chunk, 2]
        chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
        chunk_df_corrected['axis-0'] = z_values
        chunk_df_corrected['axis-1'] = y_values
        chunk_df_corrected['axis-2'] = x_values
        df = pd.concat([df, chunk_df_corrected])

    print("Saving df")
    df['index'] = list(range(df.shape[0]))
    return df


def get_chunk_indices(origin_coords, chunk_size):
    indices = []
    for origin in list(origin_coords):
        indices.append([
            slice(origin[0], origin[0] + chunk_size[0], 1),
            slice(origin[1], origin[1] + chunk_size[1], 1),
            slice(origin[2], origin[2] + chunk_size[2], 1)
        ])
    return indices


if __name__ == "__main__":
    pass
    # sys.path.append(os.getcwd())
    # from analysis.utils import split_in_chunks_nd_no_overlap, chunks_to_tiffs
    #
    # detect_cells_deepblink(sys.argv[1])

    # convert_to_napari_format("/CBI_Hive/Acquire/CEBRA/03CL12/analysis/composites_RSCM_v0.1_job_01086.ims/resolution_level_x/channel_1_chunks")

    # tiff_stack_shape = [696, 6397, 4886]
    # ratios = (np.array(tiff_stack_shape) / np.array(DEEPBLINK_CHUNK_SIZE)).astype('int') + 1
    # patchify_chunks_shape = (*list(ratios), *DEEPBLINK_CHUNK_SIZE)
    # print("patchify_chunks_shape", patchify_chunks_shape)
    # origin_coords = get_origin_coords(3, patchify_chunks_shape, DEEPBLINK_CHUNK_SIZE)
    # np.save("/CBI_Hive/Acquire/CEBRA/03CL12/analysis/composites_RSCM_v0.1_job_01086.ims/resolution_level_x/channel_1_chunks/origin_coords.npy", origin_coords)
    # merge_df("/CBI_Hive/Acquire/CEBRA/03CL12/analysis/composites_RSCM_v0.1_job_01086.ims/resolution_level_x/channel_1_chunks", origin_coords)
