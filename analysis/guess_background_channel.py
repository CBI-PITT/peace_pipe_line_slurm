from glob import glob
import json
import logging
import os

import numpy as np
from scipy import fft
from skimage.transform import resize
from skimage import metrics
import tifffile

from analysis import settings
from analysis.guess_brain_orientation import guess_orientation

log = logging.getLogger(__name__)


def read_info_file(output_folder):
    """
    Read json file that describes analysis of this brain.
    """
    info_file_path = os.path.join(output_folder, settings.INFO_FILE_NAME)
    log.debug(f"Reading dataset_info file at {info_file_path}")

    with open(info_file_path, "r") as f:
        options_str = f.read()
    options = json.loads(options_str)
    return options



def guess_by_histogram_max(ims_file):
    channels_with_smallest_histogram_max = {}
    for resolution_level in range(ims_file.ResolutionLevels):
        histogram_max_by_channel = [
            ims_file.metaData[resolution_level, 0, c, 'HistogramMax'] for c in range(ims_file.Channels)
        ]
        if min(histogram_max_by_channel) == max(histogram_max_by_channel):
            continue
        channels_with_smallest_histogram_max[resolution_level] = np.argmin(
            np.asarray(histogram_max_by_channel)
        )
    log.info("Channels with lowest HistogramMax: ", channels_with_smallest_histogram_max)
    weights = np.linspace(0.2, 0.8, num=len(channels_with_smallest_histogram_max))
    return int(np.round(
        np.average(np.asarray(list(channels_with_smallest_histogram_max.values())), weights=weights)
    ))


def guess_by_fft(ims_file):
    """
    DOES NOT WORK
    """
    high_frequency_sums = []

    for channel in range(ims_file.Channels):
        volume_100um = ims_file.get_Volume_At_Specific_Resolution(channel=channel)
        depth, height, width = volume_100um.shape
        volume_fft = fft.fftn(volume_100um) / (depth * width * height)
        volume_fft = fft.fftshift(volume_fft)
        center = [depth // 2, height // 2, width // 2]
        radius = depth // 10
        z, y, x = np.ogrid[:depth, :height, :width]
        mask_area = (x - center[2]) ** 2 + (z - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2  # sphere
        volume_fft[mask_area] = 0
        volume_fft = fft.ifftshift(volume_fft)
        high_frequency_sum = np.sum(np.real(fft.ifftn(volume_fft)) * depth * width * height)
        high_frequency_sums.append(high_frequency_sum)

    return np.argmin(np.asarray(high_frequency_sums))


def guess_by_nmi(ims_file, save_100um_volume=False, out_dir=None):
    """
    Again NMI :)
    Works as good as histogram method
    """
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas
    import bg_space as bgs

    if out_dir:
        try:
            options = read_info_file(out_dir)
        except FileNotFoundError:
            options = {}
    else:
        out_dir = ims_file.filePathBase
        options = {}
    orientation = options.get('orientation')
    if not orientation:
        orientation = guess_orientation(ims_file)
    atlas = BrainGlobeAtlas("allen_mouse_100um")
    nmi_channels = []
    for channel in range(ims_file.Channels):
        volume_100um_file = os.path.join(
            out_dir,
            f"{ims_file.fileName}_channel_{channel}{settings.SUFFIX_100UM_VOLUME}"
        )
        try:
            volume_100um = tifffile.imread(volume_100um_file)
        except FileNotFoundError:
            volume_100um = ims_file.get_Volume_At_Specific_Resolution(channel=channel)
        if save_100um_volume and not os.path.exists(volume_100um_file):
            tifffile.imwrite(volume_100um_file, volume_100um)
        reorient_volume_asr = bgs.map_stack_to(orientation, "asr", volume_100um)
        resized_reorient_volume_asr = resize(
            reorient_volume_asr,
            (atlas.reference.shape[0], atlas.reference.shape[1], atlas.reference.shape[2]),
            anti_aliasing=True
        )
        nmi = metrics.normalized_mutual_information(resized_reorient_volume_asr, atlas.reference)
        nmi_channels.append(nmi)
    return int(np.argmax(np.asarray(nmi_channels)))


def guess_background(ims_file, save_100um_volume=False, out_dir=None):
    """
    Guess background channel by one or several methods in this module.

    :param ims_file: ims class instance for which orientation should be determined
    :param save_100um_volume: bool, save 100 um volume for each channel or not
    :param out_dir: str, folder to save 100 um volume
    :return: int
    """
    log.info(f"Guessing background channel for {ims_file}:")
    if ims_file.Channels == 1:
        return 0, 0
    background_channel_hist = guess_by_histogram_max(ims_file)
    log.info(f"\t- Guess of background channel based on histogram: {background_channel_hist}")
    background_channel_nmi = guess_by_nmi(ims_file, save_100um_volume=save_100um_volume, out_dir=out_dir)
    log.info(f"\t- Guess of background channel based on NMI: {background_channel_nmi}")
    return background_channel_hist, background_channel_nmi


if __name__ == "__main__":
    folder = "/CBI_Hive/Acquire/Klimstra/03CL02/"
    ims_file_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.ims'):
                ims_file_paths.append(os.path.join(root, file))

    ims_file_paths = [x for x in ims_file_paths if not ('cropped' in x or 'scene' in x)]
    for ims_file_path in ims_file_paths:
        guess_background(ims_file_path)
