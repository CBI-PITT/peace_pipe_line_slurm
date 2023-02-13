#!/usr/bin/env python
# coding: utf-8
import glob
import logging
import os

import numpy as np
import tifffile
from skimage.filters import threshold_minimum, threshold_triangle
from skimage.measure import label, regionprops
from skimage.morphology import disk, opening
from scipy import ndimage

from analysis.contrast_stretching import stretch_contrast
from analysis.denoise import denoise_fft, denoise_fft_ellipse


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


if __name__ == "__main__":
    parent_folder = '/CBI_Hive/Public/klimstra-w/2020 - 02CL19/analysis/2CL19 48 HPI Eeev 1_00159_denoise.ims/resolution_level_3'
    output_folder = os.path.join(parent_folder, 'autofluorescence_background_subtracted')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pattern = os.path.join(parent_folder, 'autofluorescence', '*.tif')
    files = sorted(glob.glob(pattern))

    for file in files:
        print('Processing: ', file)
        image = tifffile.imread(file)
        image = stretch_contrast(image)
        image = denoise_fft(image)
        image = subtract_background(image)

        out_filename = os.path.join(output_folder, os.path.basename(file))
        tifffile.imwrite(out_filename, image.astype(np.uint16))
