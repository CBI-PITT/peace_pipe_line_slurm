import os
import sys

import ants
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
import numpy as np
import tifffile
import bg_space as bgs
from skimage.segmentation import find_boundaries


input_file = sys.argv[1]
output_folder = sys.argv[2]
atlas_name = sys.argv[3]

print("Reading atlas")
atlas = BrainGlobeAtlas(atlas_name)
print("Reading raw image")
raw = tifffile.imread(input_file)

print("Converting images to ants format")
raw_reoriented_ants = ants.from_numpy(raw.astype('float32'))
atlas_ants = ants.from_numpy(atlas.reference.astype('float32'))

print("Running registration...")
mytx = ants.registration(fixed=atlas_ants, moving=raw_reoriented_ants, type_of_transform='SyN')

registration_output_folder = output_folder

print("Saving outputs")
# save warped raw image
warped_raw_ants = mytx['warpedmovout']
warped_raw = warped_raw_ants.numpy()
tifffile.imwrite(os.path.join(registration_output_folder, 'downsampled_standard.tiff'), warped_raw)

# save warped atlas image
warped_atlas_ants = mytx['warpedfixout']
warped_atlas = warped_atlas_ants.numpy()
tifffile.imwrite(os.path.join(registration_output_folder, 'warped_atlas.tiff'), warped_atlas)

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
    os.path.join(registration_output_folder, 'registered_atlas.tiff'),
    warped_atlas_annotation
)

# save boundary image
boundaries_image = find_boundaries(warped_atlas_annotation, mode="inner").astype(
    np.int8, copy=False
)
tifffile.imwrite(os.path.join(registration_output_folder, 'boundaries.tiff'), boundaries_image)
print("All done!")
