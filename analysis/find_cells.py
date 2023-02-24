from datetime import datetime
import json
import logging
import os
import re
import subprocess

import dask
import numpy as np
import pandas as pd
import tifffile

from analysis import settings
from analysis.guess_background_channel import guess_background
from analysis.guess_brain_orientation import guess_orientation
from analysis.utils import (
    ensure_tiffs_extracted,
    find_appropriate_atlas,
    get_resolution_level_better_than_10um,
    get_signal_channels,
    read_info_file,
    update_info_file
)

log = logging.getLogger(__name__)


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


def detect_cells(options):
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas
    from imlib.IO.cells import get_cells
    from imaris_ims_file_reader import ims

    log.info('Starting cell detection ...')
    analysis_dir_this_brain = options["out_name"]
    ims_file = ims(options['ims_file_path'])  # TODO: pass ims file object instead of path str
    if not analysis_dir_this_brain:  # make analysis folder next to the ims file
        analysis_dir_this_brain = os.path.join(
            os.path.dirname(ims_file.filePathComplete),
            'analysis',
            ims_file.FileName
        )
    if not os.path.exists(analysis_dir_this_brain):
        os.makedirs(analysis_dir_this_brain)

    info_file_path = os.path.join(analysis_dir_this_brain, settings.INFO_FILE_NAME)

    if os.path.exists(info_file_path):
        old_options = read_info_file(analysis_dir_this_brain)
        options.update(old_options)
    else:
        from analysis.main import initialize_info_file
        initialize_info_file(ims_file, analysis_dir_this_brain)

    atlas = BrainGlobeAtlas(settings.ATLAS_NAME_FORMAT.format(options["allen_resolution"]))
    soma_diameter = options.get('cell_size', settings.CELLFINDER_SOMA_DIAMETER)  # um
    threshold = options.get('threshold', settings.CELLFINDER_THRESHOLD)  # sigmas above mean
    align = options.get('align', False)
    classify = options.get('classify', False)

    out_directory = os.path.join(analysis_dir_this_brain, settings.RESOLUTION_LEVEL_FOLDER_NAME)

    # Extract tiff series
    for channel in range(options["channels"]):
        ensure_tiffs_extracted(channel, options)

    signal_channels = get_signal_channels(options["channels"], options["background_channel"])
    signal_folders = []
    for signal_channel in signal_channels:
        signal_folders.append(os.path.join(out_directory, f"channel_{signal_channel}"))

    background_folder = os.path.join(out_directory, f"channel_{options['background_channel']}")
    cellfinder_output_folder = os.path.join(out_directory, settings.CELLFINDER_OUT_FOLDER_NAME)

    success = launch_cell_finder(
        signal_folders,
        background_folder,
        cellfinder_output_folder,
        options["resolution"],
        options["orientation"],
        atlas,
        threshold=threshold,
        soma_diameter=soma_diameter,
        align=align,
        classify=classify  # TODO: should we always separate detection and classification?
    )

    if not success:
        return False  # go to next operation

    log.debug(f"Saving points to npy file {os.path.join(cellfinder_output_folder, 'all_detected_spots.npy')}")
    if len(signal_channels) == 1:
        cellfinder_output_folders = [cellfinder_output_folder]
    else:
        cellfinder_output_folders = []
        for signal_channel in range(len(signal_channels)):
            cellfinder_output_folders.append(os.path.join(cellfinder_output_folder, f"channel_{signal_channel}"))

    for cellfinder_output_folder in cellfinder_output_folders:
        # get all detected spots into one array
        if options.get('classify', False):
            all_detections = get_cells(os.path.join(cellfinder_output_folder, 'points', 'cell_classification.xml'), cells_only=True)
        else:
            all_detections = get_cells(os.path.join(cellfinder_output_folder, 'points', 'cells.xml'))

        all_detected_spots = []

        for cell in all_detections:
            all_detected_spots.append([cell.z, cell.y, cell.x])

        all_detected_spots = np.asarray(all_detected_spots)
        np.save(os.path.join(cellfinder_output_folder, settings.DETECTED_SPOTS_FILE_NAME), all_detected_spots)

    log.info('Finished detecting cells')
    return True


def launch_cell_finder(
        signal_folders,
        background_folder,
        output_folder,
        resolution,
        orientation,
        atlas,
        threshold=6,
        soma_diameter=10,
        align=False,
        classify=False,
        path_to_model=None
):
    """
    Run cellfinder via subprocess.

    All parameters are described here: https://docs.brainglobe.info/cellfinder/user-guide/command-line
    """
    log.debug(f"Launching cellfinder. Saving to {output_folder}")
    cmd = [
        'cellfinder',
        '-s', *signal_folders, '-b', background_folder, '-o', output_folder,
        '-v', str(resolution[0]), str(resolution[1]), str(resolution[2]),
        '--orientation', orientation,
        '--atlas', atlas.atlas_name,
        '--no-analyse', '--no-figures',
        '--threshold', str(threshold), '--soma-diameter', str(soma_diameter), '--artifact-keep'
    ]
    if not align:
        cmd.append('--no-register')

    if not classify:
        cmd.append('--no-classification')
    elif path_to_model is not None:
        cmd.append('--trained-model')
        cmd.append(str(path_to_model).strip())

    log.debug(f"Cellfinder CMD: {cmd}")

    # TODO: how to run the 'ulimit -n 60000' command before cellfinder?
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        if ret.stdout:
            log.error(ret.stdout.decode())
        log.error('Cell detection failed')
        return False
    log.debug("Cellfinder: success")
    return True


def run_dbscan_on_chunk(df):
    from sklearn.cluster import DBSCAN
    points = df[["axis-0", "axis-1", "axis-2"]].to_numpy()
    print("Points shape", points.shape)

    eps = 3
    min_samples = 2

    # create a DBSCAN object
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # fit the points to the model
    dbscan.fit(points)

    # get the cluster assignments for each point
    labels = dbscan.labels_

    # get the unique cluster labels
    cluster_labels = np.unique(labels)
    print("Total clusters:", len(cluster_labels))

    # calculate the centroid of each cluster
    def get_cluster_centroid(label):
        points_in_cluster = points[labels == label]
        centroid = np.mean(points_in_cluster, axis=0)
        return centroid

    centroids = [dask.delayed(get_cluster_centroid)(x) for x in cluster_labels]
    cluster_centroids = dask.compute(centroids)[0]
    print(type(cluster_centroids))
    print(len(cluster_centroids))

    cluster_centroids = np.array(cluster_centroids)
    print(cluster_centroids.shape)
    print(cluster_centroids.dtype)
    df = pd.DataFrame()
    df['axis-0'] = cluster_centroids[:, 0]
    df['axis-1'] = cluster_centroids[:, 1]
    df['axis-2'] = cluster_centroids[:, 2]
    return df


def merge_detected_cells(options):
    """
    Combines deepblink and cellfinder output
    Then runs dbscan on combined output, to remove duplication
    :return:
    """
    from imlib.IO.cells import get_cells

    analysis_dir_this_brain = options["out_name"]
    signal_channels = sorted(get_signal_channels(options["channels"], options["background_channel"]))
    signal_channel = options.get("signal_channel", signal_channels[0])
    resolution_level_dir = os.path.join(analysis_dir_this_brain, settings.RESOLUTION_LEVEL_FOLDER_NAME)
    cellfinder_output_dir = os.path.join(resolution_level_dir, settings.CELLFINDER_OUT_FOLDER_NAME)
    if len(signal_channels) == 1:
        cellfinder_output_folders = [cellfinder_output_dir]
    else:
        cellfinder_output_folders = []
        for channel in range(len(signal_channels)):
            cellfinder_output_folders.append(os.path.join(cellfinder_output_dir, f"channel_{channel}"))
    deepblink_csv = os.path.join(resolution_level_dir, "output_deepblink", f"channel_{signal_channel}_cells.csv")
    cellfinder_npy = os.path.join(cellfinder_output_dir, "all_detected_spots.npy")
    merged_csv_path = os.path.join(cellfinder_output_dir, "points", "combined_cells_cellfinder_deepblink.csv")
    dbscan_csv_path = os.path.join(cellfinder_output_dir, "points", "combined_cells_cellfinder_deepblink_dbscan_3_2.csv")

    def restore_cellfinder_points():
        points_dir = cellfinder_output_folders[signal_channels.index(signal_channel)]
        all_detections = get_cells(os.path.join(points_dir, 'points', 'cells.xml'))
        all_detected_spots = []

        for cell in all_detections:
            all_detected_spots.append([cell.z, cell.y, cell.x])

        all_detected_spots = np.asarray(all_detected_spots)
        np.save(os.path.join(points_dir, settings.DETECTED_SPOTS_FILE_NAME), all_detected_spots)
        np.save(cellfinder_npy, all_detected_spots)

    print("Restoring cellfinder points")
    restore_cellfinder_points()
    print("Restored")
    df = pd.read_csv(deepblink_csv)
    deepblink_points = df[['axis-0', 'axis-1', 'axis-2']].to_numpy()
    cellfinder_points = np.load(cellfinder_npy)
    merged_points = np.concatenate([deepblink_points, cellfinder_points])
    print("merged_points", merged_points.shape)

    merged_df = pd.DataFrame()
    merged_df['axis-0'] = merged_points[:,0]
    merged_df['axis-1'] = merged_points[:,1]
    merged_df['axis-2'] = merged_points[:,2]
    merged_df.to_csv(merged_csv_path)

    dbscan_df = run_dbscan_on_chunk(merged_df)
    dbscan_df.to_csv(dbscan_csv_path)
    try:
        os.rename(cellfinder_npy, os.path.join(os.path.dirname(cellfinder_npy), f"cellfinder_{os.path.basename(cellfinder_npy)}"))
    except:
        pass
    points = dbscan_df[['axis-0', 'axis-1', 'axis-2']].to_numpy()
    np.save(cellfinder_npy, points)
    return True


def main(input_file_path, output_file_path, *args):
    """
    Count cells using cellfinder, register to the atlas (optional) and save detailed cell info to csv (optional).

    CBPy compatible

    :param input_file_path: path to a file with a list of ims files and corresponding parameters for cell counting.
    format of the input file (JSON compatible):
    [
    {
        "ims_file_path": "path/to/file1.ims",
        "cell_size": 10,
        "orientation": "sal",
        "background_channel": 0,
        "align": true,
        "allen_resolution": 10,
        "classify": false,
    },
    {
        "ims_file_path": "path/to/file2.ims",
        "cell_size": 10,
        "orientation": "sal",
        "background_channel": 0,
        "align": true,
        "allen_resolution": 10,
        "classify": false,
    },
    ...
    ]
    :param output_file_path: path to the output csv with detailed cell info
    :param args: whatever
    :return: None
    """
    with open(input_file_path, 'r') as f:
        input_data_str = f.read()
    # pattern = re.compile(r'\s+')
    # input_data_str = re.sub(pattern, '', input_data_str)
    input_data_str = input_data_str.replace("'", '"')
    input_data = json.loads(input_data_str)
    for data_entry in input_data:
        data_entry['out_filename'] = output_file_path
        detect_cells(data_entry)


if __name__ == "__main__":
    main(
        '/CBI_FastStore/Public/iana/hooks_brain_a_cellfinder/hooks_find_cells.txt',
        '/CBI_FastStore/Public/iana/hooks_brain_a_cellfinder/cells_detailed_info.csv'
    )
