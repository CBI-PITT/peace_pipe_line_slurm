#!/usr/bin/env python
# coding: utf-8
from glob import glob
import logging
import os

import numpy as np
import pandas as pd
import tifffile
from skimage.filters import threshold_minimum, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening
from scipy import ndimage

from analysis.contrast_stretching import stretch_contrast
from analysis.denoise import denoise_fft, denoise_fft_ellipse
from analysis import settings


def subtract_background(img):
    import cv2

    img = img.astype(np.float32)
    try:
        thr = threshold_minimum(img)
    except:
        try:
            thr = threshold_triangle(img)
            thr = thr - 0.15 * thr  # TODO hardcoded value
        except ValueError:  # attempt to get argmax of an empty sequence
            return np.zeros_like(img).astype(np.uint16)

    mask = (img > thr).astype(np.uint8)
    mask = (ndimage.binary_fill_holes(mask)).astype(np.uint8)
    kernel = disk(3)
    mask = (opening(mask, kernel)).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1
    for i in range(nb_components):
        if sizes[i] < 10000:  # TODO: hardcoded value (try to retain N largest components).
            mask[output == i + 1] = np.abs(mask[output == i + 1] - 1)

    img[mask == 0] = 0
    return img.astype(np.uint16)


def subtract_background_coronal_plane(img):
    """
    Subtract background from downsampled_standard.tif image, to compute overlap and nmi with atlas.

    Mask is computed from denoised image with low cutoff frequency
    (blured signigicantly) for better thresholding.
    """
    import cv2

    img_lf = denoise_fft_ellipse(img)
    img_lf = img_lf.astype(np.float32)
    try:
        thr = threshold_triangle(img_lf)
    except ValueError:  # attempt to get argmax of an empty sequence
        return np.zeros_like(img).astype(np.uint16)

    mask = (img_lf > thr).astype(np.uint8)
    mask = (ndimage.binary_fill_holes(mask)).astype(np.uint8)
    kernel = disk(3)
    mask = (opening(mask, kernel)).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    nb_components -= 1
    for i in range(nb_components):
        if sizes[i] < 10000:  # TODO: hardcoded value (try to retain N largest components).
            mask[output == i + 1] = np.abs(mask[output == i + 1] - 1)

    img[mask == 0] = 0
    return img


def subtract_background_iterative(img, min_percent_zeros=20, max_percent_zeros=90, mask_only=False):
    # TODO rewrite as recursion (for readability)
    # TODO move constants to settings
    """
    Compute foreground/background mask by iteratively changing the threshold.

    :param img: image plane (z layer)
    :return: mask (0s and 1s)
    """

    def compute_mask(img, thr):
        mask = (img > thr).astype(np.uint8)
        mask = (ndimage.binary_fill_holes(mask)).astype(np.uint8)
        kernel = disk(3)
        mask = (opening(mask, kernel)).astype(np.uint8)
        return mask

    max_iterations = 100
    max_components = 5
    threshold_step = 0.15

    img_pixels = img.shape[0] * img.shape[1]
    percent0 = 0
    img = img.astype(np.float32)
    # TODO denoise?
    try:
        threshold = threshold_triangle(img)
    except ValueError:  # attempt to get argmax of an empty sequence
        return np.zeros_like(img).astype(np.uint8)

    iteration = 0
    while percent0 <= min_percent_zeros:
        # Increase threshold
        if iteration > max_iterations:
            return np.zeros_like(img)
        threshold = threshold + threshold_step * threshold
        mask = None
        mask = compute_mask(img, threshold)
        percent0 = (mask[mask == 0].shape[0]) / img_pixels * 100
        iteration += 1

    while percent0 >= max_percent_zeros:
        # Decrease threshold
        if iteration > max_iterations:
            return np.zeros_like(img)
        threshold = threshold - threshold_step * threshold
        mask = None
        mask = compute_mask(img, threshold)
        percent0 = (mask[mask == 0].shape[0]) / img_pixels * 100
        iteration += 1

    label_image = label(mask)
    rp = regionprops(label_image)
    rp = sorted(rp, key=lambda x: x.area, reverse=True)
    final_mask = np.zeros_like(mask)
    for i in range(min([max_components, len(rp)])):
        for c in rp[i].coords:
            final_mask[c[0], c[1]] = 1
    img[final_mask == 0] = 0
    if mask_only:
        return final_mask
    return img.astype(np.uint16)


# def remove_background_detections(options):
#     """
#     Remove background detections after deepblink.
#     pros: The dataframe format can be left as is
#     cons: Doesn't get rid of cellfinder's bg detections
#
#     :param options:
#     :return:
#     """
#     from cellfinder_core.tools.IO import read_with_dask
#
#     # we have chunks folder
#     signal_channel = get_signal_channels()[0]
#     chunks_folder = os.path.join(options['out_name'], settings.RESOLUTION_LEVEL_FOLDER_NAME, f'channel_{signal_channel}_chunks')
#     # we have mask saved as tiled tiffs
#     mask_folder = os.path.join(options['out_name'], settings.RESOLUTION_LEVEL_FOLDER_NAME, RANDOM_FOREST_MASK_FOLDER_NAME)
#     # we read all mask as dask array
#     lazy_mask_stack = read_with_dask(mask_folder)
#
#     origin_coords = np.load(os.path.join(chunks_folder, 'origin_coords.npy'))
#
#     # we read each chunk
#     def read_chunk(chunk_number):
#         return tifffile.imread(os.path.join(chunks_folder, f"chunk_{str(chunk_number).zfill(5)}.tif"))
#
#     # we read corresponding area from the mask
#     def read_mask(chunk_number):
#         origin = origin_coords[chunk_number]
#         end = np.array(origin) + np.array(settings.DEEPBLINK_CHUNK_SIZE)
#         mask = lazy_mask_stack[origin[0]: end[0], origin[1]: end[1], origin[2]: end[2]]
#         return mask.compute()
#
#     # remove points
#     def process_chunk(chunk, mask):
#         return points
#
#     # save a new df for each chunk
#     def save_df():
#         pass
#
#     # merge new df
#
#     df = pd.read_csv()
#     points = df[['axis-0', 'axis-1', 'axis-2']].to_numpy()
#     detected_cells_np = np.floor(points).astype(int)
#     cells_binary = np.zeros(nuclei_chunk_shape, dtype=np.uint8)
#     print("Converting to binary", cells_binary.shape)
#     np.put(cells_binary, np.ravel_multi_index(detected_cells_np.T, nuclei_chunk_shape), 1)
#     print("Multiplying by mask")
#     mask_folder = os.path.join(str(Path(spectral_info_folder).parent.parent), 'scale_4', 'mask_resized')
#     mask_stack = tifffile.imread(os.path.join(mask_folder, f"chunk_{str(number).zfill(5)}.tif"))
#     mask_stack = resize(mask_stack, nuclei_chunk_shape)
#     cells_filtered = cells_binary * mask_stack
#     print("Converting to coords")
#     nz = np.nonzero(cells_filtered)
#     zipped_nz = list(zip(*nz))
#     filtered_cells_np = np.asarray(zipped_nz)
#     print("filtered_cells_np", filtered_cells_np.shape)
#     print("Generating csv")
#     filtered_cells_df = pd.DataFrame()
#     filtered_cells_df['axis-0'] = list(filtered_cells_np[:, 0])
#     filtered_cells_df['axis-1'] = list(filtered_cells_np[:, 1])
#     filtered_cells_df['axis-2'] = list(filtered_cells_np[:, 2])
#     print("Saving coords to csv")
#     filtered_cells_df.to_csv(os.path.join(dbscan_folder, f"filtered_chunk_{str(number).zfill(5)}.csv"))
#     return True


def segment_background_apoc(options):
    """
    Run random forest pixel classification to separate bg and fg.

    :param options:
    :return:
    """
    from apoc import PixelClassifier
    model_path = os.path.join(options['out_name'], settings.RESOLUTION_LEVEL_FOLDER_NAME, "PixelClassifier.cl")
    segmenter = PixelClassifier(opencl_filename=model_path)
    input_folder = os.path.join(options['out_name'], settings.RESOLUTION_LEVEL_FOLDER_NAME, f"channel_{options['background_channel']}")
    output_folder = os.path.join(options['out_name'], settings.RESOLUTION_LEVEL_FOLDER_NAME, settings.RANDOM_FOREST_MASK_FOLDER_NAME)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def get_mask(file_name):
        raw_data = tifffile.imread(file_name)
        result = segmenter.predict(image=raw_data)
        np_result = np.array(result).astype(np.uint8)
        return np_result == 2

    filenames = sorted(glob(os.path.join(input_folder, "*.tif")))

    for z, fname in enumerate(filenames):
        mask = get_mask(fname)
        tifffile.imwrite(os.path.join(output_folder, f'{str(z).zfill(5)}.tif'), mask)


def remove_background_detections(options):
    """
    Remove false positive cell detections in the background.

    :param options:
    :return:
    """
    cellfinder_output_folder = os.path.join(
        options['out_name'],
        settings.RESOLUTION_LEVEL_FOLDER_NAME,
        settings.CELLFINDER_OUT_FOLDER_NAME
    )
    path_to_classification_df = os.path.join(
        cellfinder_output_folder,
        'points',
        f'predictions_{settings.PYTORCH_MODEL_NAME}_{settings.PYTORCH_MODEL_VERSION}.csv'
    )
    df = pd.read_csv(path_to_classification_df)
    combined_filtered_cells_df = []
    combined_filtered_non_cells_df = []
    mask_folder = os.path.join(options['out_name'], settings.RESOLUTION_LEVEL_FOLDER_NAME, settings.RANDOM_FOREST_MASK_FOLDER_NAME)
    save_results_to = os.path.join(cellfinder_output_folder, 'points')

    def process(df_part):
        print(df_part.head())
        points = df_part[['axis-1', 'axis-2']].to_numpy()
        print("Points", points.shape)
        detected_cells_np = np.floor(points).astype(int)
        # create a binary image where points are represented as pixels with value 1, the rest of the image is 0
        cells_binary = np.zeros(options['shape'][1:], dtype=np.uint8)
        print("cells_binary", cells_binary.shape)
        np.put(cells_binary, np.ravel_multi_index(detected_cells_np.T, options['shape'][1:]), 1)
        # read the mask
        # multiply cells image by mask
        cells_filtered = cells_binary * mask
        # get indices of points that get removed, "unravel" them
        # removed = cells_binary[np.logical_and(cells_binary == 1, mask == 0)]
        # convert binary image back to points
        nz = np.nonzero(cells_filtered)
        zipped_nz = list(zip(*nz))
        filtered_cells_np = np.asarray(zipped_nz)
        print("filtered_cells_np", filtered_cells_np.shape)
        # save points as dataframe
        filtered_cells_df = pd.DataFrame()
        if filtered_cells_np.shape[0]:
            filtered_cells_df['axis-0'] = [z] * filtered_cells_np.shape[0]
            filtered_cells_df['axis-1'] = list(filtered_cells_np[:, 0])
            filtered_cells_df['axis-2'] = list(filtered_cells_np[:, 1])
        return filtered_cells_df

    # loop through z
    for z in range(options['shape'][0]):
        print("Processing: ", z)
        # select all points at current z
        partial_df = df[df["axis-0"] == z]
        partial_df_cells = partial_df[partial_df["nn_decoded"] == 'cell']
        partial_df_non_cells = partial_df[partial_df["nn_decoded"] == 'non_cell']
        mask = tifffile.imread(os.path.join(mask_folder, f"{str(z).zfill(5)}.tif"))

        filtered_cells_df = pd.DataFrame()
        filtered_non_cells_df = pd.DataFrame()
        if partial_df_cells.shape[0]:
            filtered_cells_df = process(partial_df_cells)
        if partial_df_non_cells.shape[0]:
            filtered_non_cells_df = process(partial_df_non_cells)

        # append remaining points to a new dataframe
        combined_filtered_cells_df.append(filtered_cells_df)
        combined_filtered_non_cells_df.append(filtered_non_cells_df)

    print("Saving dfs to csv")
    combined_filtered_cells_df = pd.concat(combined_filtered_cells_df, ignore_index=True)
    combined_filtered_cells_df.to_csv(
        os.path.join(
            save_results_to,
            f'predicted_cells_{settings.PYTORCH_MODEL_NAME}_{settings.PYTORCH_MODEL_VERSION}_bg_removed.csv'
        )
    )
    combined_filtered_non_cells_df = pd.concat(combined_filtered_non_cells_df, ignore_index=True)
    combined_filtered_non_cells_df.to_csv(
        os.path.join(
            save_results_to,
            f'predicted_non_cells_{settings.PYTORCH_MODEL_NAME}_{settings.PYTORCH_MODEL_VERSION}_bg_removed.csv'
        )
    )
    combined_filtered_cells_df['nn_decoded'] = ['cell'] * combined_filtered_cells_df.shape[0]
    combined_filtered_non_cells_df['nn_decoded'] = ['non_cell'] * combined_filtered_non_cells_df.shape[0]
    combined_cells_noncells = pd.concat([combined_filtered_cells_df, combined_filtered_non_cells_df], ignore_index=True)
    combined_cells_noncells.to_csv(
        os.path.join(
            cellfinder_output_folder,
            'points',
            f'predictions_{settings.PYTORCH_MODEL_NAME}_{settings.PYTORCH_MODEL_VERSION}_bg_removed.csv'
        )
    )
    return True
