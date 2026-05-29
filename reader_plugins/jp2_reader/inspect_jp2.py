import json
import sys

import glymur


def main():
    input_path = sys.argv[1]
    jp2_wrapper = glymur.Jp2k(input_path)
    raw_shape = list(jp2_wrapper.shape)

    if len(raw_shape) == 2:
        yx_shape = raw_shape
    elif len(raw_shape) == 3:
        raise ValueError(
            'JP2 reader expects a folder of 2D grayscale JP2 slices. '
            'Volumetric or color JP2 files are not supported.'
        )
    else:
        raise ValueError(f'Unsupported JP2 shape: {raw_shape}')

    print(json.dumps({'raw_shape': raw_shape, 'shape': yx_shape}))


if __name__ == '__main__':
    main()
