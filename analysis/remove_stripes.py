from collections import defaultdict
from glob import glob
import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tifffile

from analysis import settings
from analysis.utils import read_info_file

log = logging.getLogger(__name__)


def fft_1d_stripes_filter(image):
    first_harmonic = get_first_harmonic_from_globals()

    harmonics = image.shape[1] // first_harmonic
    ind1 = np.arange(harmonics // 2 + 1) * first_harmonic
    ind_dilated_3 = []

    for i in ind1[1:]:
        ind_dilated_3.extend([i - 1, i, i + 1])

    if ind_dilated_3[-1] >= image.shape[1]:
        ind_dilated_3.pop()

    for i in range(image.shape[0]):
        input_arr = image[i, :].astype(np.float32)
        img_fft = np.fft.fft(input_arr) / input_arr.shape[0]
        img_fft[ind_dilated_3] = 0
        inverse_fft = np.fft.ifft(img_fft)
        image[i, :] = (np.real(inverse_fft) * input_arr.shape[0]).astype(np.uint16)

    return image


def calculate_first_harmonic_one_img(image, stripes_direction='v'):
    """
    Calculate as argmax().

    stripes_period = int(np.floor(1./first_harmonic*len(quant_5)))
    """
    if stripes_direction == "h":
        axis = 1
    elif stripes_direction == "v":
        axis = 0
    else:
        raise NotImplemented("Can only automatically remove vertical or horizontal stripes")
    quant_5 = np.quantile(image.astype(np.float32), 0.05, axis=axis)
    r_fft_transf = np.fft.rfft(quant_5)/quant_5.shape[0]
    first_harmonic = np.argmax(abs(r_fft_transf)[5:]) + 5
    log.info(f"Calculated first harmonic: {first_harmonic}")
    return first_harmonic


def get_first_harmonic_from_globals():
    from analysis.pre_process import FIRST_HARMONIC
    if not FIRST_HARMONIC:
        from analysis.register_brains import FIRST_HARMONIC
    return FIRST_HARMONIC


def calculate_first_harmonic_from_stitching(ims_file, analysis_dir_this_brain):
    """
    Calculate from stitching info (shiftValues.csv, stitchData.csv). Only works for RSCM (obviously).
    """
    pth = Path(ims_file.filePathComplete)
    composites_dir = pth.parent.parent
    shift_values_csv = str(composites_dir / 'shiftValues.csv')
    overlap_csv = str(composites_dir / 'stitchData.csv')
    if not os.path.exists(shift_values_csv) or not os.path.exists(overlap_csv):
        log.warning("Stitching csv files do not exist")
        return
    shift_df = pd.read_csv(shift_values_csv)
    x_shift = shift_df.iloc[0]['xShift']
    overlap_df = pd.read_csv(str(overlap_csv))
    overlap = overlap_df.iloc[0]['overlap']
    stripes_width_full_resolution = 1024 - overlap + x_shift
    dataset_info = read_info_file(analysis_dir_this_brain)
    stripes_width_resolution_x = stripes_width_full_resolution * dataset_info['shape'][-1] / ims_file.shape[-1]
    calculated_first_harmonic = int(np.floor(dataset_info['shape'][-1] / stripes_width_resolution_x))
    return calculated_first_harmonic


def calculate_first_harmonic_from_majority(analysis_dir_this_brain):
    """
    Compute first harmonic of the striped artifact from the image.

    Computes first harmonics for each z layer (2D slice), then returns the most
    frequent value among them as the true first harmonic.
    :param analysis_dir_this_brain: str
    :return: int
    """
    log.info("Computing 1st harmonic from the data")
    dataset_info = read_info_file(analysis_dir_this_brain)
    harmonics = defaultdict(int)
    imgs = sorted(glob(
        os.path.join(
            analysis_dir_this_brain,
            settings.RESOLUTION_LEVEL_FOLDER_NAME,
            f'channel_{dataset_info["background_channel"]}',
            '*.tif'
        )
    ))
    for z, img_name in enumerate(imgs):
        img = tifffile.imread(img_name)
        first_harmonic = calculate_first_harmonic_one_img(img)
        log.debug(f"z layer {z}: first harmonic {first_harmonic}")
        harmonics[first_harmonic] += 1

    harmonics = dict(harmonics)
    most_frequent = max(harmonics, key=harmonics.get)
    return int(most_frequent)


def ideal_notch_filter(fshift, points):
    d0 = 121.0  # cutoff frequency
    H, W = fshift.shape
    u, v = np.ogrid[:H, :W]
    for d in range(len(points)):
        u0, v0 = points[d]
        mask1 = (u - u0)**2 + (v - v0)**2 <= d0
        mask2 = (u + u0)**2 + (v + v0)**2 <= d0
        fshift[mask1] = 0
        fshift[mask2] = 0
    return fshift


def gaussian_notch_filter(fft_shift_img, points):
    d0 = 77  # cutoff frequency
    H, W = fft_shift_img.shape
    u, v = np.ogrid[:H, :W]
    for d in range(len(points)):
        u0, v0 = points[d]
        d1 = ((u - u0)**2 + (v - v0)**2)**0.5
        d2 = ((u + u0)**2 + (v + v0)**2)**0.5
        fft_shift_img[u,v] *= (1 - np.exp(-0.5 * (d1 * d2 / d0**2)))
    return fft_shift_img


def fft_2d_notch_filter(image, stripes_direction="v"):
    """
    stripes_direction: "h" (horizontal stripes) or "v" (vertical stripes)
    """
    log.debug("Starting fft_2d_notch_filter")
    image_dtype = image.dtype
    image = image.astype(np.float32)
    first_harmonic = get_first_harmonic_from_globals()
    if not first_harmonic:
        log.warning("Could not get first harmonic from stitching. Computing from image")
        first_harmonic = calculate_first_harmonic_one_img(image)
    H, W = image.shape

    # do 2D fft
    img_fft = np.fft.fft2(image) / (W * H)
    # shift fft to get maximum at the center
    img_fft = np.fft.fftshift(img_fft)

    # filter shifted fft
    points = []
    image_center = (H // 2, W // 2)
    if stripes_direction == "h":
        # compute points on vertical axis of symmetry
        for point_ind in range(1, image_center[0] // first_harmonic):
            points.extend([
                [image_center[0] + point_ind * first_harmonic, image_center[1]],
                [image_center[0] - point_ind * first_harmonic, image_center[1]]
            ])
    elif stripes_direction == "v":
        # compute points on horizontal axis of symmetry
        for point_ind in range(1, image_center[1] // first_harmonic):
            points.extend([
                [image_center[0], image_center[1] + point_ind * first_harmonic],
                [image_center[0], image_center[1] - point_ind * first_harmonic]
            ])
    else:
        raise NotImplemented("Can only automatically remove vertical or horizontal stripes")
    points = np.asarray(points)

    tst = time.time()
    filtered_fft_shift = ideal_notch_filter(img_fft, points)
    tfi = time.time()

    # unshift
    img_fft = np.fft.ifftshift(filtered_fft_shift)
    # do inverse fft
    out_ifft = np.fft.ifft2(img_fft)
    # image = (np.real(out_ifft) * W * H).astype(image_dtype)
    image = (np.abs(out_ifft) * W * H).astype(image_dtype)  # works better for fMOST data
    return image
