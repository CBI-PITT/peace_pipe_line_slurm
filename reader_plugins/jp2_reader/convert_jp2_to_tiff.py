import os
import sys
from glob import glob

import glymur
import numpy as np
import tifffile


def get_output_file(output_dir, channel, z_index):
    return os.path.join(
        output_dir,
        f'resolution_level_0',
        f'channel_{channel}',
        f'r00_t00_c00_z{str(z_index).zfill(4)}.tif'
    )


def get_rgb_output_file(output_dir, z_index):
    return os.path.join(
        output_dir,
        'resolution_level_0',
        'rgb_color',
        f'r00_t00_c00_z{str(z_index).zfill(4)}.tif'
    )


def to_uint8(data):
    if data.dtype == np.uint8:
        return data

    data = np.asarray(data)
    data_min = data.min()
    data_max = data.max()

    if data_max == data_min:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = (data.astype(np.float32) - float(data_min)) / float(data_max - data_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


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
        tifffile.imwrite(get_output_file(output_dir, 0, z_index), to_uint8(data))
    elif data.ndim == 3 and data.shape[-1] == 3:
        rgb_channels = []
        for channel in range(3):
            channel_data = to_uint8(data[..., channel])
            rgb_channels.append(channel_data)
            tifffile.imwrite(get_output_file(output_dir, channel, z_index), channel_data)
        tifffile.imwrite(get_rgb_output_file(output_dir, z_index), np.stack(rgb_channels, axis=-1))
    elif data.ndim == 3:
        raise ValueError(
            'JP2 reader expects a folder of 2D grayscale or RGB JP2 slices. '
            'Volumetric JP2 files are not supported.'
        )
    else:
        raise ValueError(f'Unsupported JP2 shape: {data.shape}')


if __name__ == '__main__':
    main()
