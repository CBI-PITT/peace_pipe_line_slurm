from glob import glob
import logging
import os
from pathlib import Path
import shutil
from datetime import datetime

from imaris_ims_file_reader import ims
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate

from analysis import settings
from analysis.utils import read_info_file, update_info_file, get_resolution_level_better_than_10um

log = logging.getLogger(__name__)


def classify_cells_fastai(options):
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
    # path = Path(options["out_name"])
    # analysis_folder = path.parent.absolute()
    model_path = settings.PYTORCH_MODEL_PATH
    model_version = settings.PYTORCH_MODEL_VERSION
    model_name = os.path.basename(model_path)
    # dataset_info_file = os.path.join(options["out_name"], 'dataset_info.json')

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

    options_update = read_info_file(options['out_name'])
    options.update(options_update)

    points_to_classify_npy = os.path.join(options["out_name"], 'all_detected_spots_imaris.npy')
    if not os.path.exists(points_to_classify_npy):
        log.error(f"No imaris points npy file: {points_to_classify_npy}")
        return False
    save_results_to = os.path.join(cellfinder_output_folder, 'points')
    if os.path.exists(
        os.path.join(
            save_results_to,
            f'predicted_non_cells_{model_name}_{model_version}.csv'
        )
    ) and os.path.exists(
        os.path.join(
            save_results_to,
            f'predicted_cells_{model_name}_{model_version}.csv'
        )
    ):
        return True

    tst = datetime.now()
    # Transform npy file to napari-compatible csv
    points = np.load(points_to_classify_npy)
    log.info(f"-----------Total points {points.shape[0]} -----------")
    points_df = pd.DataFrame()
    z_values = np.round(points[:, 0]).astype(int)
    y_values = np.round(points[:, 1]).astype(int)
    x_values = np.round(points[:, 2]).astype(int)
    points_df['index'] = list(range(points.shape[0]))
    points_df['axis-0'] = z_values
    points_df['axis-1'] = y_values
    points_df['axis-2'] = x_values
    points_csv_name = os.path.join(os.path.dirname(points_to_classify_npy),
                                   os.path.basename(points_to_classify_npy).replace('.npy', '_napari.csv'))
    points_df.to_csv(points_csv_name)

    df_inference = points_df
    df_inference.rename(columns={'axis-0': 'axis_0', 'axis-1': 'axis_1', 'axis-2': 'axis_2'}, inplace=True)
    n_cubes = df_inference.shape[0]
    df_inference['ann'] = ['unknown1'] * (n_cubes // 2) + ['unknown2'] * (n_cubes - n_cubes // 2)
    data_sel = df_inference

    # with open(dataset_info_file, "r") as f:
    #     settings_str = f.read()
    #     settings = json.loads(settings_str)

    resolution_level = options['resolution_level']
    ims_file = ims(options['ims_file_path'], ResolutionLevelLock=resolution_level)

    # Extract cubes the same way it is done in cellfinder
    cube_shape = (20, 50, 50)
    cube_shape2 = (20, 25, 25)
    img_shape = ims_file.metaData[resolution_level, 0, 0, 'shape'][-3:]

    def get_cube_slicing2(z, y, x):
        """
        Extract cube like cellfinder does. Subsequent zoom needed.
        """
        z_left = max([z - cube_shape2[0] // 2, 0])
        z_right = min([z + cube_shape2[0] // 2, img_shape[0]])
        y_left = max([y - cube_shape2[1] // 2, 0])
        y_right = min([y_left + cube_shape2[1], img_shape[1]])
        x_left = max([x - cube_shape2[2] // 2, 0])
        x_right = min([x_left + cube_shape2[2], img_shape[2]])
        return slice(z_left, z_right, None), slice(y_left, y_right, None), slice(x_left, x_right, None)

    # create data loader
    print("Creating data loader")

    def get_x(r):
        """
        Extract cubes as cellfinder does, with zoom
        """
        slice_z, slice_y, slice_x = get_cube_slicing2(r.axis_0, r.axis_1, r.axis_2)
        raw_cube = np.array(ims_file[resolution_level, 0, 0, slice_z, slice_y, slice_x])
        zoomed_cube = zoom(raw_cube, [1, 2, 2], order=2)
        return zoomed_cube

    def get_y(r):
        return r['ann']

    def int2float(o: TensorImage):
        return o.float().div_(2 ** 16)

    class ImageND(Tuple):
        @classmethod
        def create(cls, nd_image):
            nd_image = nd_image.astype(float)
            nd_image = tensor(nd_image)

            return nd_image

    def ImageNDBlock():
        return TransformBlock(type_tfms=ImageND.create,
                              batch_tfms=int2float)

    dblock = DataBlock(blocks=(ImageNDBlock, CategoryBlock),
                       get_x=get_x,
                       get_y=get_y,
                       splitter=RandomSplitter(valid_pct=0.1))

    dsets = dblock.datasets(data_sel)
    dls = dblock.dataloaders(data_sel)

    # create a learner
    print("Loading model")
    learn = vision_learner(dls, resnet50, metrics=error_rate)
    nChannels = 20
    learn.model[0][0] = nn.Conv2d(nChannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # load model
    learn.load(model_path)

    # run inference
    points_limit = 500000
    parts = n_cubes // points_limit + 1
    prob = []
    decoded = []
    incomplete_cubes = []
    for part in range(parts):
        start_row = part * points_limit
        end_row = min([(part + 1) * points_limit -1, n_cubes])
        partial_df = df_inference.loc[start_row: end_row]
        print(f"Gathering inference cubes, part {part} of {parts}")
        test_files = (get_x(x) for i, x in partial_df.iterrows())
        incomplete = []  # some points are at the edges of the image, so cubes are incomplete
        complete_test_files = []
        for f in test_files:
            complete = f.shape == cube_shape
            incomplete.append(not complete)
            if complete:
                complete_test_files.append(f)
        # partial_df['incomplete'] = incomplete
        incomplete_cubes.extend(incomplete)
        test_dl = learn.dls.test_dl(complete_test_files)

        print(f"Running inference, part {part} of {parts}")
        preds, _, decoded_values = learn.get_preds(dl=test_dl, with_decoded=True)
        probabilities = [float(x[0]) for x in preds]
        prob.extend(probabilities)
        decoded.extend(decoded_values)

    # save output
    print("Saving outputs")
    v = ["cell", "non_cell"]
    decoded_values = [v[x] for x in decoded]
    df_inference['incomplete'] = incomplete_cubes
    df_inference_complete = df_inference.loc[df_inference.incomplete == False].copy()
    df_inference_complete['nn_decoded'] = decoded_values
    df_inference_complete['prob'] = prob
    df_inference_complete.to_csv(
        os.path.join(
            save_results_to,
            f'predictions_{model_name}_{model_version}.csv'
        )
    )

    df_inference_cells = df_inference_complete.loc[df_inference_complete["nn_decoded"] == "cell"]
    df_inference_non_cells = df_inference_complete.loc[df_inference_complete["nn_decoded"] == "non_cell"]

    # save classification results to napari compatible csv
    cells_df = pd.DataFrame()
    cells_df['index'] = list(range(df_inference_cells.shape[0]))
    cells_df['axis-0'] = df_inference_cells.axis_0.to_list()
    cells_df['axis-1'] = df_inference_cells.axis_1.to_list()
    cells_df['axis-2'] = df_inference_cells.axis_2.to_list()
    cells_df.to_csv(
        os.path.join(
            save_results_to,
            f'predicted_cells_{model_name}_{model_version}.csv'
        )
    )

    non_cells_df = pd.DataFrame()
    non_cells_df['index'] = list(range(df_inference_non_cells.shape[0]))
    non_cells_df['axis-0'] = df_inference_non_cells.axis_0.to_list()
    non_cells_df['axis-1'] = df_inference_non_cells.axis_1.to_list()
    non_cells_df['axis-2'] = df_inference_non_cells.axis_2.to_list()
    non_cells_df.to_csv(
        os.path.join(
            save_results_to,
            f'predicted_non_cells_{model_name}_{model_version}.csv'
        )
    )
    tfi = datetime.now()
    log.info(f"------------ Classification took {tfi - tst} -------------")
    return True
