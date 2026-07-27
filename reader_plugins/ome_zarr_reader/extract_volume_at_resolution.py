# conda activate omehans-reader
import json
import os
import sys
from pathlib import Path

import tifffile
import zarr
from dask import array as da
from skimage import img_as_float32
from skimage.transform import rescale

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f'Running on {os.uname().nodename}')


def extract_volume_at_resolution(channel=0, output_resolution=(100, 100, 100)):
    metadata_path = os.path.join(input_dir, '.zattrs')
    if os.path.exists(metadata_path):
        metadata = json.load(open(metadata_path, 'r'))
    else:
        metadata = dict(zarr.open(input_dir, mode='r').attrs)

    resolution_levels = len(metadata['multiscales'][0]['datasets'])
    full_resolution = metadata['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'][-3:]
    resolutionLevelToExtract = 0
    for res in range(resolution_levels):
        currentResolution = metadata['multiscales'][0]['datasets'][res]['coordinateTransformations'][0]['scale'][-3:]
        resCompare = [x <= y for x, y in zip(currentResolution, output_resolution)]
        resEqual = [x == y for x, y in zip(currentResolution, full_resolution)]
        if all(resCompare) == True or (all(resCompare) == False and any(resEqual) == True):
            resolutionLevelToExtract = res

    workingVolumeResolution = metadata['multiscales'][0]['datasets'][resolutionLevelToExtract]['coordinateTransformations'][0]['scale'][-3:]
    print('Reading ResolutionLevel {}'.format(resolutionLevelToExtract))

    if os.path.exists(os.path.join(input_dir, f'scale{resolutionLevelToExtract}')):
        scale_dir = f'scale{resolutionLevelToExtract}'
    elif os.path.exists(os.path.join(input_dir, f's{resolutionLevelToExtract}')):
        scale_dir = f's{resolutionLevelToExtract}'
    else:
        scale_dir = f'{resolutionLevelToExtract}'

    root = zarr.open(input_dir, mode='r')
    zarray = root[scale_dir]
    dask_zarray = da.array(zarray)

    workingVolume = dask_zarray[0, channel, :, :, :].compute()
    print('type of workingVolume', type(workingVolume))
    print('shape of workingVolume', workingVolume.shape)

    print('Resizing volume from resolution in microns {} to {}'.format(str(workingVolumeResolution), str(output_resolution)))
    rescaleFactor = tuple([round(x / y, 5) for x, y in zip(workingVolumeResolution, output_resolution)])
    print('Rescale Factor = {}'.format(rescaleFactor))

    workingVolume = img_as_float32(workingVolume)
    workingVolume = rescale(workingVolume, rescaleFactor, anti_aliasing=True)
    return workingVolume


input_dir = sys.argv[1]
output_path = sys.argv[2]
channel = int(sys.argv[3])

volume = extract_volume_at_resolution(channel=channel)
tifffile.imwrite(output_path, volume)
