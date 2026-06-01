import json
import sys

import glymur


def get_mode_and_shape(raw_shape):
    if len(raw_shape) == 2:
        return 'grayscale', 1, raw_shape

    if len(raw_shape) == 3 and raw_shape[-1] == 3:
        return 'rgb', 3, raw_shape[:2]

    if len(raw_shape) == 3:
        raise ValueError(
            'JP2 reader expects a folder of 2D grayscale or RGB JP2 slices. '
            'Volumetric JP2 files are not supported.'
        )

    raise ValueError(f'Unsupported JP2 shape: {raw_shape}')


def main():
    input_path = sys.argv[1]
    jp2_wrapper = glymur.Jp2k(input_path)
    raw_shape = list(jp2_wrapper.shape)
    mode, channels, yx_shape = get_mode_and_shape(raw_shape)

    print(json.dumps({
        'raw_shape': raw_shape,
        'shape': yx_shape,
        'mode': mode,
        'channels': channels
    }))


if __name__ == '__main__':
    main()
