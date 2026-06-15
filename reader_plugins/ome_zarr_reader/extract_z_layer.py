# conda activate omehans-reader
import os
import sys
from pathlib import Path

import tifffile
import zarr
from dask import array as da

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f'Running on {os.uname().nodename}')


input_dir = sys.argv[1]
output_dir = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
z = int(sys.argv[5])


root = zarr.open(input_dir, mode='r')
zarray = root[f'scale{resolution_level}']
dask_zarray = da.array(zarray)

plane = dask_zarray[0, channel, z, :, :].compute()

tifffile.imwrite(os.path.join(
    output_dir,
    f'resolution_level_{str(resolution_level)}',
    f'channel_{str(channel)}',
    f'r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif'
), plane, tile=(512, 512), compression='zlib')
