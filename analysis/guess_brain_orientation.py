import os
import logging
import glob

import tifffile
from skimage.transform import resize
from skimage import metrics

from analysis import settings

log = logging.getLogger(__name__)


def guess_orientation(ims_file, save_100um_volume=False, out_dir=None):
    """
    Determine which of brain orientations ('sal' or 'spr') is more likely.

    Uses Normalized Mutual Information between the reoriented data stack
    (downsampled to 100 um) and the 100 um atlas.
    :param ims_file: ims class instance for which orientation should be determined
    :param save_100um_volume: bool, save 100 um volume for channel 0 or not
    :param out_dir: str, folder to save 100 um volume
    :return: str ('sal' or 'spr')
    """
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas
    import bg_space as bgs

    log.info(f"Guessing orientation for {ims_file.filePathComplete}")
    atlas = BrainGlobeAtlas("allen_mouse_100um")
    volume_100um = ims_file.get_Volume_At_Specific_Resolution()  # determine for channel 0
    if save_100um_volume and out_dir:  # Save 100um volume to specified directory
        tifffile.imwrite(
            os.path.join(out_dir, f"{ims_file.fileName}_channel_0{settings.SUFFIX_100UM_VOLUME}"),
            volume_100um
        )
    elif save_100um_volume:  # No directory specified to save 100um volume, save next to ims file
        tifffile.imwrite(
            os.path.join(ims_file.filePathBase, f"{ims_file.fileName}_channel_0{settings.SUFFIX_100UM_VOLUME}"),
            volume_100um
        )
    reorient_spr_asr = bgs.map_stack_to("spr", "asr", volume_100um)
    resized_reorient_spr_asr = resize(
        reorient_spr_asr,
        (atlas.reference.shape[0], atlas.reference.shape[1], atlas.reference.shape[2]),
        anti_aliasing=True
    )
    nmi_spr = metrics.normalized_mutual_information(resized_reorient_spr_asr, atlas.reference)
    reorient_sal_asr = bgs.map_stack_to("sal", "asr", volume_100um)
    resized_reorient_sal_asr = resize(
        reorient_sal_asr,
        (atlas.reference.shape[0], atlas.reference.shape[1], atlas.reference.shape[2]),
        anti_aliasing=True
    )
    nmi_sal = metrics.normalized_mutual_information(resized_reorient_sal_asr, atlas.reference)
    log.info(f"\tNMI for spr: {nmi_spr}\n\tNMI for sal: {nmi_sal}")
    guessed_orientation = 'spr' if nmi_spr > nmi_sal else 'sal'
    log.info(f"Guessed orientation for {ims_file.filePathComplete} is {guessed_orientation}")
    return guessed_orientation


if __name__ == "__main__":
    folder = "/CBI_Hive/Public/klimstra-w/2019 - 01CL73/"
    ims_file_paths = glob.glob(folder + "*.ims")
    print(f"Total ims files: {len(ims_file_paths)}")
    for ims_file_path in ims_file_paths:
        orientation = guess_orientation(ims_file_path)
        print(os.path.basename(ims_file_path), orientation)
