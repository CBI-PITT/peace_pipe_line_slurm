import os
import sys
from glob import glob

import glymur
import tifffile


def get_output_file(output_dir, z_index):
    return os.path.join(output_dir, f'r00_t00_c00_z{str(z_index).zfill(4)}.tif')


def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    z_index = int(sys.argv[3])

    os.makedirs(output_dir, exist_ok=True)

    jp2_files = glob(os.path.join(input_dir, '*.jp2'))
    jp2_files.extend(glob(os.path.join(input_dir, '*.JP2')))
    jp2_files = sorted(jp2_files)
    input_path = jp2_files[z_index]

    jp2_wrapper = glymur.Jp2k(input_path)
    data = jp2_wrapper[:]

    if data.ndim == 2:
        tifffile.imwrite(get_output_file(output_dir, z_index), data)
    elif data.ndim == 3:
        raise ValueError(
            'JP2 reader expects a folder of 2D grayscale JP2 slices. '
            'Volumetric or color JP2 files are not supported.'
        )
    else:
        raise ValueError(f'Unsupported JP2 shape: {data.shape}')


if __name__ == '__main__':
    main()
