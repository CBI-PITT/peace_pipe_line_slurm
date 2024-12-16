import os
import sys

import tifffile
import zarr
from dask import array as da
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store


input_dir = sys.argv[1]
output_dir = sys.argv[2]
resolution_level = int(sys.argv[3])
channel = int(sys.argv[4])
z = int(sys.argv[5])


location = os.path.join(input_dir, f'scale{resolution_level}')
store = H5_Nested_Store(location)
zarray = zarr.open(store)
dask_zarray = da.array(zarray)

plane = dask_zarray[0, channel, z, :, :]

tifffile.imwrite(os.path.join(
    output_dir,
    f"resolution_level_{str(resolution_level)}",
    f"channel_{str(channel)}",
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
), plane)
