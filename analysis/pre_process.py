from glob import glob
import logging
import os
import sys

import tifffile
from dask.delayed import delayed
import dask

from analysis.contrast_stretching import stretch_contrast, stretch_contrast_and_denoise
from analysis.background_subtraction import subtract_background, subtract_background_iterative
from analysis.denoise import denoise_fft
from analysis.remove_stripes import fft_1d_stripes_filter, fft_2d_notch_filter
from analysis import settings
from analysis.utils import ensure_tiffs_extracted, read_info_file, update_info_file


log = logging.getLogger(__name__)


FIRST_HARMONIC = None

"""
Pre-processing variants that we have:
    - contrast stretching (standalone or combined with other methods)
    - background subtraction (standalone or combined with other methods)
    - denoising by FFT (standalone or combined with other methods)
    - 1D FFT stripes removal (standalone or combined with other methods)
    - 1D FFT stripes removal with restriction of bright signal
    - 2D FFT stripes removal (standalone or combined with other methods)
    - 2D FFT denoise
    - 2D FFT notch filter (ideal or gaussian)

Automation options (thoughts):
    PSNR -> do we need FFT denoising?
    FFT of 5% quantile -> 1st harmonic intensity -> do we need to remove stripes?

Current pre-processing:
    - contrast stretching
    - contrast stretching + background subtraction
    - 1D FFT
    - 1D FFT + contrast stretching
    - 1D FFT + contrast stretching + background subtraction
    - 2D FFT denoise
    - 2D FFT notch filter (ideal or gaussian)
total: 7 folders

"""


def pre_process_brain_faster(parent_folder, channel=0, use_dask=False):
    method_sequences = settings.PREPROCESSING_METHODS  # Ex: [ ['fft_2d_notch_filter'], ['fft_2d_notch_filter', 'stretch_contrast'], ['fft_2d_notch_filter', 'stretch_contrast', 'subtract_background']]
    dir_prefixes = []
    # finished_methods = []
    for method_sequence in method_sequences:
        previous_method_prefixes = []
        for ind, method in enumerate(method_sequence):
            prefix = f'channel_{channel}{"_" if ind > 0 else ""}{"_".join(previous_method_prefixes)}'
            folder = os.path.join(parent_folder, prefix)
            method_prefix = settings.PREPROCESSING_METHOD_PREFIX_MAP[method]
            # if method not in finished_methods:
            if use_dask:
                apply_method_parallel(method, folder)
            else:
                apply_method(method, folder)
            # finished_methods.append(method)
            previous_method_prefixes.append(method_prefix)
            dir_prefixes.append(prefix + "_" + method_prefix)

    dir_prefixes.append(f'channel_{channel}{"_" if ind > 0 else ""}{"_".join(previous_method_prefixes)}')  # loop variables should still be there after the loop finished
    return dir_prefixes


def apply_method(method_name, img_folder):
    method = getattr(sys.modules[__name__], method_name)
    out_folder = img_folder + '_' + settings.PREPROCESSING_METHOD_PREFIX_MAP[method_name]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    files = sorted(glob(os.path.join(img_folder, "*.tif")))
    for file in files:
        out_filename = os.path.join(out_folder, os.path.basename(file))
        if os.path.exists(out_filename):
            log.warning(f"Warning: {out_filename} exists.")
            continue
        img = tifffile.imread(file)
        img_processed = method(img)
        tifffile.imwrite(out_filename, img_processed)


def apply_method_parallel(method_name, img_folder):
    method = getattr(sys.modules[__name__], method_name)
    out_folder = img_folder + '_' + settings.PREPROCESSING_METHOD_PREFIX_MAP[method_name]

    def read_file(in_path):
        print(f"Reading file {in_path}")
        return tifffile.imread(in_path)

    def process_file(img, preproc_method):
        print(f"Processing file", preproc_method.__name__)
        return preproc_method(img)

    def write_file(img, f):
        out_path = os.path.join(out_folder, os.path.basename(f))
        print(f"Writing file {out_path}")
        return tifffile.imwrite(out_path, img)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    files = sorted(glob(os.path.join(img_folder, "*.tif")))
    files = [f for f in files if not os.path.exists(os.path.join(out_folder, os.path.basename(f)))]

    imgs = [delayed(read_file)(f) for f in files]
    processed = [delayed(process_file)(img, method) for img in imgs]
    saved = [delayed(write_file)(img, f) for f, img in zip(files, processed)]
    saved = dask.compute(saved)


def pre_process(options):
    """
    Wrapper to call pre-processing standalone
    :param options: dict
    :return: bool
    """
    from imaris_ims_file_reader import ims

    parent_folder = os.path.join(options['out_name'], "resolution_level_x")
    ims_file = ims(options["ims_file_path"])
    retrieve_fft_first_harmonic(options)
    use_dask = os.uname().nodename in settings.DASK_ALLOWED_NODES
    for channel in range(ims_file.Channels):
        ensure_tiffs_extracted(channel, options)
        pre_process_brain_faster(parent_folder, channel=channel, use_dask=use_dask)


def retrieve_fft_first_harmonic(options):
    """
    Set global FIRST_HARMONIC variable to be shared with fft related functions.

    Try to get from dataset info file.
    If failed, try to compute from stitching info.
    If not available, compute from image (compute for each z layer, then select most frequent value).
    :param options: dict
    :return: None
    """
    from analysis.remove_stripes import calculate_first_harmonic_from_stitching, calculate_first_harmonic_from_majority
    from imaris_ims_file_reader import ims

    ims_file = ims(options["ims_file_path"])
    analysis_dir_this_brain = options['out_name']
    dataset_info = read_info_file(analysis_dir_this_brain)

    global FIRST_HARMONIC
    try:
        FIRST_HARMONIC = dataset_info["first_harmonic"]
    except KeyError:
        FIRST_HARMONIC = calculate_first_harmonic_from_stitching(ims_file, analysis_dir_this_brain)
        dataset_info['first_harmonic'] = FIRST_HARMONIC
        update_info_file(analysis_dir_this_brain, dataset_info)

    if not FIRST_HARMONIC:
        FIRST_HARMONIC = calculate_first_harmonic_from_majority(analysis_dir_this_brain)
        dataset_info['first_harmonic'] = FIRST_HARMONIC
        update_info_file(analysis_dir_this_brain, dataset_info)
