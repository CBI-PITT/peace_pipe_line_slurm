from collections import defaultdict
from glob import glob
import json
import logging
import os

import dask
import dask.array as da
from patchify import patchify, unpatchify
import numpy as np
import tifffile
from numpy.lib.stride_tricks import as_strided


from analysis import settings

log = logging.getLogger(__name__)


def create_info_file(output_folder, options):
    """
    Create json file that describes analysis of this brain.

    Stores:
     guessed orientation
     guessed background channel
     highest resolution atlas NiftyReg can be used with (usually 10 or 25um)
     TODO: PSNR for each channel
     TODO: folder with best registration
     TODO: overlap betweeen warped brain and atlas

    :param output_folder: str, path to the analysis output folder for this brain
    :param options: dict, analysis parameters
    :return: None
    """
    log.debug(f"Creating dataset_info file at {os.path.join(output_folder, settings.INFO_FILE_NAME)}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, settings.INFO_FILE_NAME), "w") as f:
        f.write(json.dumps(options))


def update_info_file(output_folder, options):
    """
    Update json file that describes analysis of this brain.

    Stores:
     guessed orientation
     guessed background channel
     highest resolution atlas NiftyReg can be used with (usually 10 or 25um)
     TODO: PSNR for each channel
     TODO: folder with best registration
     TODO: overlap betweeen warped brain and atlas

    :param output_folder: str, path to the analysis output folder for this brain
    :param options: dict, analysis parameters
    :return: None
    """
    info_file_path = os.path.join(output_folder, settings.INFO_FILE_NAME)
    log.debug(f"Updating dataset_info file at {info_file_path}")

    if not os.path.exists(info_file_path):
        create_info_file(output_folder, options)

    with open(info_file_path, "r") as f:
        options_str = f.read()
    old_options = json.loads(options_str)
    old_options.update(options)
    with open(info_file_path, "w") as f:
        f.write(json.dumps(old_options))


def read_info_file(output_folder):
    """
    Read json file that describes analysis of this brain.

    Stores:
     guessed orientation
     guessed background channel
     highest resolution atlas NiftyReg can be used with (usually 10 or 25um)
     TODO: PSNR for each channel
     TODO: folder with best registration
     TODO: overlap betweeen warped brain and atlas

    :param output_folder: str, path to the analysis output folder for this brain
    :return: dict
    """
    info_file_path = os.path.join(output_folder, settings.INFO_FILE_NAME)
    log.debug(f"Reading dataset_info file at {info_file_path}")

    with open(info_file_path, "r") as f:
        options_str = f.read()
    options = json.loads(options_str)
    return options


def get_resolution_level_better_than_10um(ims_file):
    resolution_level = ims_file.ResolutionLevels - 1
    highest_resolution = ims_file.resolution

    while resolution_level >= 0:
        resolution = ims_file.metaData[resolution_level, 0, 0, 'resolution']
        if resolution[0] == highest_resolution[0] and resolution[1] < 10:  # TODO doesn't work for fmost
            break
        resolution_level -= 1

    shape = ims_file.metaData[resolution_level, 0, 0, 'shape'][-3:]

    return resolution_level, resolution, shape


def find_appropriate_atlas(ims_file):
    allen_resolutions = [100, 50, 25, 10]

    data_shape = ims_file.shape[-3:]
    highest_resolution = ims_file.resolution
    data_shape_um = np.asarray(data_shape) * np.asarray(highest_resolution)
    opimal_atlas = None
    for allen_res in allen_resolutions:
        if (data_shape_um/allen_res).max() < 2048:
            opimal_atlas = allen_res

    return opimal_atlas


def get_signal_channels(number_of_channels, background_channel):
    if number_of_channels == 1:  # no background
        signal_channels = [0]
    elif number_of_channels == 2:  # 1 signal channel, 1 background
        signal_channels = [int(not background_channel)]
    else:  # multiple signal channels = other folder structure
        signal_channels = [x for x in range(number_of_channels) if x != background_channel]
    return signal_channels


def create_registration_info_file(output_folder):
    registration_info_file_path = os.path.join(output_folder, settings.REGISTRATION_INFO_FILE_NAME)
    log.debug(f"Creating registration_info file at {registration_info_file_path}")
    with open(registration_info_file_path, "w") as f:
        f.write(json.dumps({}))


def read_registration_info_file(output_folder):
    registration_info_file_path = os.path.join(output_folder, settings.REGISTRATION_INFO_FILE_NAME)
    log.debug(f"Reading registration_info file at {registration_info_file_path}")

    if not os.path.exists(registration_info_file_path):
        create_registration_info_file(output_folder)

    with open(registration_info_file_path, "r") as f:
        info_str = f.read()
    info = json.loads(info_str)
    return info


def update_registration_info_file(output_folder, info):
    registration_info_file_path = os.path.join(output_folder, settings.REGISTRATION_INFO_FILE_NAME)
    log.debug(f"Updating registration_info file at {registration_info_file_path}")

    if not os.path.exists(registration_info_file_path):
        create_registration_info_file(output_folder)

    with open(registration_info_file_path, "r") as f:
        options_str = f.read()
    old_options = json.loads(options_str)
    old_options.update(info)
    with open(registration_info_file_path, "w") as f:
        f.write(json.dumps(old_options))


def read_in_progress_files_json():
    in_progress_ims_files = {}
    with open(settings.IN_PROGRESS_IMS_LOCATION, 'r') as f:
        input_data_str = f.read()
        in_progress_data = json.loads(input_data_str)
        in_progress_ims_files = defaultdict(list)
        for fl, op in in_progress_data.values():
            in_progress_ims_files[fl].append(op)
        in_progress_ims_files = dict(in_progress_ims_files)
    return in_progress_ims_files


def update_in_progress_files_json(host, ims_file_path=None, operation=None, remove=False):
    input_data_dict = {}
    with open(settings.IN_PROGRESS_IMS_LOCATION, 'r') as f:
        input_data_str = f.read()
        input_data_dict = json.loads(input_data_str)

    if remove:
        input_data_dict[host] = [None, None]
    else:  # add
        input_data_dict[host] = [ims_file_path, operation]

    with open(settings.IN_PROGRESS_IMS_LOCATION, "w") as f:
        f.write(json.dumps(dict(input_data_dict)))


def ensure_tiffs_extracted(channel, options):
    from imaris_ims_file_reader import ims

    parent_folder = os.path.join(options['out_name'], "resolution_level_x")
    tiff_series_dir = os.path.join(parent_folder, f"channel_{channel}")
    extracted_tiffs = glob(os.path.join(tiff_series_dir, '*.tif'))
    if len(extracted_tiffs) < options['shape'][0]:
        log.info(f'Extracting tiff series to {tiff_series_dir} ...')
        ims_file = ims(options["ims_file_path"])
        ims_file.save_Tiff_Series(
            location=tiff_series_dir,
            channels=(channel,),
            resolutionLevel=options["resolution_level"],
            overwrite=True
        )


def tiff_series_to_ram(folder, img_shape, img_dtype):
    log.info("Reading tiff series to RAM")
    files = sorted(glob(os.path.join(folder, '*.tif')))
    if len(files) < img_shape[0]:
        log.warning(f"Incomplete tiff series at {folder}. Skipping.")
        return
    tiffstack = np.empty(img_shape, dtype=img_dtype)

    def read_img(img):
        print("Reading", img)
        return tifffile.imread(img)

    def save_img(z, img):
        print("Saving", z)
        tiffstack[z, :, :] = img
        return True

    imgs = [dask.delayed(read_img)(i) for i in files]
    saved = [dask.delayed(save_img)(z, i) for z, i in enumerate(imgs)]
    saved = dask.compute(saved)
    return tiffstack


def chunks_to_tiffs(chunks, output_folder):
    """
    Save chunks (result of patchify) to disk.

    :param chunks:
    :param output_folder:
    :return:
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for ind, chunk in enumerate(list(chunks)):
        chunk_file = os.path.join(output_folder, f"chunk_{str(ind).zfill(5)}.tif")
        if not os.path.exists(chunk_file):
            tifffile.imwrite(chunk_file, chunk)


def dask_chunks_to_tiffs(lazy_tiff_stack, chunk_indices, output_folder):
    for ind, slices in enumerate(chunk_indices):
        chunk_file = os.path.join(output_folder, f"chunk_{str(ind).zfill(5)}.tif")
        if not os.path.exists(chunk_file):
            chunk = lazy_tiff_stack[tuple(slices)]
            print(chunk.shape)
            chunk = chunk.compute()
            tifffile.imwrite(chunk_file, chunk)


def patchify_fn(arr_in, window_shape, step):
    """
    Function borrowed from patchify, modified to accept n-dimensional step parameter.

    :param arr_in:
    :param window_shape:
    :param step:
    :return:
    """

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (
        (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
    ) + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


def get_patchify_shape(arr_shape, window_shape, step):
    window_shape = np.array(window_shape, dtype=np.array(arr_shape).dtype)
    win_indices_shape = (
                                (np.array(arr_shape) - np.array(window_shape)) // np.array(step)
                        ) + 1
    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    return new_shape


def split_in_chunks_nd_no_overlap(img, chunk_shape):
    ndim = len(img.shape)
    ratios = (np.array(img.shape) / np.array(chunk_shape)).astype(int)
    ratios += 1
    larger_img = np.zeros((ratios * chunk_shape), img.dtype)
    if ndim == 2:  # TODO rewrite
        larger_img[:img.shape[0], :img.shape[1]] = img
    elif ndim == 3:
        larger_img[:img.shape[0], :img.shape[1], :img.shape[2]] = img
    patches = patchify_fn(larger_img, chunk_shape, step=chunk_shape)
    patches_shape = np.array(patches.shape)
    new_patches_shape = (np.prod(patches_shape[:ndim]), *patches_shape[ndim:])
    return patches.reshape(new_patches_shape)


def merge_chunks_nd_no_overlap(chunks, img_shape, chunk_shape):
    ndim = len(img_shape)
    ratios = (np.array(img_shape) / np.array(chunk_shape)).astype(int)
    ratios += 1
    old_chunks_shape = (*ratios, *chunk_shape)
    chunks = np.reshape(chunks, old_chunks_shape)
    img = unpatchify(chunks, (ratios * chunk_shape))
    if ndim == 2:  # TODO rewrite
        img = img[:img_shape[0], :img_shape[1]]
    elif ndim == 3:
        img = img[:img_shape[0], :img_shape[1], :img_shape[2]]
    return img


# def split_in_chunks_2d(img, chunk_size):
#     """
#     Split 2D image in squared chunks (chunk_size x chunk_size) without overlap.
#
#     Fill edges with zeros to make image shape multiple of chunk_size.
#     :param img:
#     :param chunk_size:
#     :return:
#     """
#     h, w = img.shape
#     print("h", h, "w", w)
#     n_patches_h = h // chunk_size + 1
#     n_patches_w = w // chunk_size + 1
#     larger_img = np.zeros((n_patches_h * chunk_size, n_patches_w * chunk_size), img.dtype)
#     larger_img[:h, :w] = img
#     print("New h, w", larger_img.shape)
#     patches = patchify(larger_img, (chunk_size, chunk_size), step=chunk_size)
#     patches = np.reshape(patches, (n_patches_h * n_patches_w, chunk_size, chunk_size))
#     return patches
#
#
# def merge_chunks_2d(chunks, chunk_size, original_img_shape):
#     """
#     Merge squared chunks obtained by split_in_chunks_2d back to a 2d image of shape original_img_shape.
#
#     :param chunks:
#     :param chunk_size:
#     :param original_img_shape:
#     :return:
#     """
#     h, w = original_img_shape
#     n_patches_h = h // chunk_size + 1
#     n_patches_w = w // chunk_size + 1
#     extended_img_shape = (n_patches_h * chunk_size, n_patches_w * chunk_size)
#     chunks = np.reshape(chunks, (n_patches_h, n_patches_w, chunk_size, chunk_size))
#     recovered_img = unpatchify(chunks, extended_img_shape)
#     recovered_img = recovered_img[:h, :w]
#     return recovered_img


# def split_in_chunks_nd_with_overlap(img, chunk_shape, overlap):
#     chunk_shape = np.array(chunk_shape)
#     chunk_shape_plus_overlap = chunk_shape + 2 * overlap
#     patches = patchify_fn(img, chunk_shape_plus_overlap, step=chunk_shape)
#     return patches
