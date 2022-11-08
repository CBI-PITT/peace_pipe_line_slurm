#!/usr/bin/env python
# coding: utf-8

import glob
import logging
import os
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage import exposure, restoration
import bg_space as bgs
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from cellfinder.analyse.analyse import transform_points_to_downsampled_space
from cellfinder.main import get_downsampled_space
from imlib.IO.cells import get_cells
from imaris_ims_file_reader import ims

from analysis.denoise import denoise_fft


def stretch_contrast(image):
    image = image.astype(np.float32)
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    image = (image * 65535).astype(np.uint16)
    return image


def stretch_contrast_and_denoise(image):
    # FFT denoising is usually done together with contrast stretching, so it makes sense to join them
    image = denoise_fft(image)
    image = stretch_contrast(image)
    return image


if __name__ == "__main__":
    parent_folder = '/CBI_FastStore/Public/iana/cebra/brain_stack1_job_00828.ims'
    output_folder = os.path.join(parent_folder, 'autofluorescence_contrast_stretched_filtered')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pattern = os.path.join(parent_folder, 'autofluorescence', '*.tif')
    files = sorted(glob.glob(pattern))

    for file in files:
        print('Processing: ', file)
        image = tifffile.imread(file)
        image = stretch_contrast(image)
        image = denoise_fft(image)
        tifffile.imwrite(out_filename, image)
