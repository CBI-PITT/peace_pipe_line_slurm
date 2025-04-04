import json
import os
import sys
from glob import glob
from pathlib import Path

import ants
import dask
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
import numpy as np
import tifffile
import bg_space as bgs
from skimage.segmentation import find_boundaries
from skimage.transform import rescale
from dask import array as da

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)


input_folder = sys.argv[1]
output_folder = sys.argv[2]
atlas_name = sys.argv[3]
orientation = sys.argv[4]

INFO_FILE_NAME = 'dataset_info.json'

print("Reading atlas")
atlas = BrainGlobeAtlas(atlas_name)

metadata = json.load(open(os.path.join(input_folder, f'.{INFO_FILE_NAME}'), 'r'))

print("Reading raw image")
if metadata["source"].endswith('.ims'):
    from imaris_ims_file_reader import ims

    ims_file = ims(metadata["source"])
    raw = ims_file[metadata['resolution_level'], 0, metadata['channel'], :, :, :]  # TODO: extract the atlas resolution directly
else:
    def read_image(file_path):
        return tifffile.imread(file_path)

    image_files = sorted(glob(os.path.join(input_folder, '*.tif*')))
    z, y, x = metadata['shape']
    lazy_arrays = [
        da.from_delayed(dask.delayed(read_image)(f), shape=(y, x), dtype='uint16')
        for f in image_files
    ]
    dask_array = da.stack(lazy_arrays, axis=0)  # Shape: (z, y, x)
    raw = dask_array.compute()  # TODO rescale individual z slices first

print("RAW shape", raw.shape)

# TODO assuming that atlas is isotropic. Should be more general.
raw_rescaled = rescale(raw, tuple(current / target for current, target in zip(metadata['resolution'], atlas.resolution)))
print("RAW rescaled shape", raw_rescaled.shape)

raw_rescaled_reoriented = bgs.map_stack_to(orientation, atlas.orientation, raw_rescaled).astype('float32')
print("RAW reoriented rescaled shape", raw_rescaled_reoriented.shape)

tifffile.imwrite(os.path.join(output_folder, 'downsampled.tiff'), raw_rescaled_reoriented)

print("Converting images to ants format")
raw_reoriented_ants = ants.from_numpy(raw.astype('float32'))
atlas_ants = ants.from_numpy(atlas.reference.astype('float32'))

print("Running registration...")
mytx = ants.registration(fixed=atlas_ants, moving=raw_reoriented_ants, type_of_transform='SyN')

print("Saving outputs")
# save warped raw image
warped_raw_ants = mytx['warpedmovout']
warped_raw = warped_raw_ants.numpy()
tifffile.imwrite(os.path.join(output_folder, 'downsampled_standard.tiff'), warped_raw)

# save warped atlas image
warped_atlas_ants = mytx['warpedfixout']
warped_atlas = warped_atlas_ants.numpy()
tifffile.imwrite(os.path.join(output_folder, 'warped_atlas.tiff'), warped_atlas)

# save warped annotation image
atlas_annotation_ants = ants.from_numpy(atlas.annotation)
warped_atlas_annotation_ants = ants.apply_transforms(
    fixed=raw_reoriented_ants,
    moving=atlas_annotation_ants,
    transformlist=mytx['invtransforms'],
    interpolator='genericLabel'
)
warped_atlas_annotation = warped_atlas_annotation_ants.numpy().astype('uint32')
tifffile.imwrite(
    os.path.join(output_folder, 'registered_atlas.tiff'),
    warped_atlas_annotation
)

# save boundary image
boundaries_image = find_boundaries(warped_atlas_annotation, mode="inner").astype(
    np.int8, copy=False
)
tifffile.imwrite(os.path.join(output_folder, 'boundaries.tiff'), boundaries_image)
print("All done!")
