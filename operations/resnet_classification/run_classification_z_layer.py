import json
import os
import sys
from glob import glob
from pathlib import Path

import tifffile
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from fastai.data.all import CategoryBlock, DataBlock, RandomSplitter, TensorImage, TransformBlock, Tuple, tensor, nn
from fastai.vision.all import vision_learner, resnet50
from fastai.metrics import error_rate
from imaris_ims_file_reader import ims

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)


input_dir = sys.argv[1]
output_dir = sys.argv[2]
box_size_str = sys.argv[3]
df_path = sys.argv[4]
model_path = sys.argv[5]
z_center = int(sys.argv[6])

box_size = list(map(int, box_size_str.split(',')))

output_path = os.path.join(output_dir, "partial_dfs")
try:
    os.makedirs(output_path)
except:
    pass

if os.path.exists(os.path.join(output_path, f'part_classification_df_z_{str(z_center).zfill(5)}.csv')):
    sys.exit(0)


def get_cube_slicing2(z, y, x):
    """
    Extract cube like cellfinder does. Subsequent zoom needed.
    """
    cube_shape2 = (20, 25, 25)
    z_left = max([z - cube_shape2[0] // 2, 0])
    z_right = min([z + cube_shape2[0] // 2, img_shape[0]])
    y_left = max([y - cube_shape2[1] // 2, 0])
    y_right = min([y_left + cube_shape2[1], img_shape[1]])
    x_left = max([x - cube_shape2[2] // 2, 0])
    x_right = min([x_left + cube_shape2[2], img_shape[2]])
    return slice(z_left, z_right, None), slice(y_left, y_right, None), slice(x_left, x_right, None)


def get_x(r):
    """
    Extract cubes as cellfinder does, with zoom
    """
    slice_z, slice_y, slice_x = get_cube_slicing2(int(round(r['axis-0'])) - z_start, int(round(r['axis-1'])), int(round(r['axis-2'])))
    raw_cube = img[slice_z, slice_y, slice_x]  # todo: img[:, slice_y, slice_x] ?
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


def read_tiff_stack(filenames, z_start, z_end):
    """
    Read a stack of TIFF files into a 3D numpy array.
    """
    img = np.expand_dims(tifffile.imread(filenames[z_start:z_end][0]), axis=0)
    print("2d img", img.shape)
    for filename in filenames[z_start:z_end][1:]:
        img = np.concatenate((img, np.expand_dims(tifffile.imread(filename), axis=0)), axis=0)
    print("3d img", img.shape)
    return img


def extract_boxes(img, partial_df, z_start):
    """
    Extract boxes of a specified size from the 3D numpy array.
    """
    def get_x(r):
        """
        Extract cubes as cellfinder does, with zoom
        """
        slice_z, slice_y, slice_x = get_cube_slicing2(int(round(r['axis-0'])) - z_start, int(round(r['axis-1'])), int(round(r['axis-2'])))
        raw_cube = img[slice_z, slice_y, slice_x]  # todo: img[:, slice_y, slice_x] ?
        zoomed_cube = zoom(raw_cube, [1, 2, 2], order=2)
        return zoomed_cube

    test_files = [get_x(x) for i, x in partial_df.iterrows()]
    print(len(test_files))

    return test_files

metadata_path = os.path.join(input_dir, '.dataset_info.json')
metadata = json.load(open(metadata_path, 'r'))
df = pd.read_csv(df_path)
df = df.round().astype(int)

img_shape = metadata['shape']
z_start = z_center - box_size[0] // 2
z_end = z_start + box_size[0]
print("z start, z end, z center", z_start, z_end, z_center)
partial_df = df[df["axis-0"] == z_center]
print("partial_df", partial_df.shape)
partial_df_complete = partial_df.loc[
    (partial_df['axis-1'] >= box_size[1] // 2)
    & (partial_df['axis-1'] <= img_shape[1] - box_size[1] // 2 - 1)
    & (partial_df['axis-2'] >= box_size[2] // 2)
    & (partial_df['axis-2'] <= img_shape[2] - box_size[2] // 2 - 1)
].copy()
print("partial_df_complete", partial_df_complete.shape)
partial_df_incomplete = partial_df.loc[
    (partial_df['axis-1'] < box_size[1] // 2)
    | (partial_df['axis-1'] > img_shape[1] - box_size[1] // 2 - 1)
    | (partial_df['axis-2'] < box_size[2] // 2)
    | (partial_df['axis-2'] > img_shape[2] - box_size[2] // 2 - 1)
    ].copy()
print("partial_df_incomplete", partial_df_incomplete.shape)

filenames = sorted(glob(os.path.join(input_dir, '*.tif')))
img = read_tiff_stack(filenames, z_start, z_end)  # TODO could be large; use dask
boxes = extract_boxes(img, partial_df_complete, z_start)
print("boxes", len(boxes))

if len(boxes):
    # create data loader
    dblock = DataBlock(blocks=(ImageNDBlock, CategoryBlock),
                       get_x=get_x,
                       get_y=get_y,
                       splitter=RandomSplitter(valid_pct=0.1))

    n_cubes = partial_df.shape[0]
    df_inference = partial_df.copy()
    df_inference['ann'] = ['unknown1'] * (n_cubes // 2) + ['unknown2'] * (n_cubes - n_cubes // 2)
    dls = dblock.dataloaders(df_inference)

    # create a learner
    print("Loading model")
    learn = vision_learner(dls, resnet50, metrics=error_rate)
    nChannels = 20
    learn.model[0][0] = nn.Conv2d(nChannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # load model
    learn.load(model_path.replace(".pth", ""))

    test_dl = learn.dls.test_dl(boxes)
    preds, _, decoded_values = learn.get_preds(dl=test_dl, with_decoded=True)
    probabilities = [float(x[0]) for x in preds]
    v = ["cell", "non_cell"]
    decoded_values = [v[x] for x in decoded_values]
    partial_df_complete['nn_decoded'] = decoded_values
    partial_df_complete['prob'] = probabilities
    partial_df_complete['incomplete'] = [False] * partial_df_complete.shape[0]

partial_df_incomplete['nn_decoded'] = [''] * partial_df_incomplete.shape[0]
partial_df_incomplete['prob'] = [np.nan] * partial_df_incomplete.shape[0]
partial_df_incomplete['incomplete'] = [True] * partial_df_incomplete.shape[0]
partial_df = pd.concat([partial_df_complete, partial_df_incomplete], ignore_index=True)
partial_df.to_csv(os.path.join(output_path, f'part_classification_df_z_{str(z_center).zfill(5)}.csv'))
