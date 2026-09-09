import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)
print(f"Running on {os.uname().nodename}")


def cast_result(result, dtype1, save_as_float):
    if save_as_float:
        return result.astype(np.float32)
    if np.issubdtype(dtype1, np.integer):
        info = np.iinfo(dtype1)
        return np.clip(result, info.min, info.max).astype(dtype1)
    return result.astype(dtype1)


def apply_bitwise_or_logical(image1, image2, calculator_operation, dtype1):
    if np.issubdtype(dtype1, np.integer):
        image1_int = image1.astype(dtype1, copy=False)
        if calculator_operation == 'not':
            return np.bitwise_not(image1_int)
        image2_int = image2.astype(dtype1, copy=False)
        if calculator_operation == 'and':
            return np.bitwise_and(image1_int, image2_int)
        if calculator_operation == 'or':
            return np.bitwise_or(image1_int, image2_int)
        return np.bitwise_xor(image1_int, image2_int)

    image1_bool = image1 != 0
    if calculator_operation == 'not':
        return np.logical_not(image1_bool)
    image2_bool = image2 != 0
    if calculator_operation == 'and':
        return np.logical_and(image1_bool, image2_bool)
    if calculator_operation == 'or':
        return np.logical_or(image1_bool, image2_bool)
    return np.logical_xor(image1_bool, image2_bool)


def get_dtype_max(dtype):
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def apply_operation(image1, image2, calculator_operation, save_as_float):
    dtype1 = image1.dtype

    if calculator_operation in {'and', 'or', 'xor', 'not'}:
        result = apply_bitwise_or_logical(image1, image2, calculator_operation, dtype1)
        return cast_result(result, dtype1, save_as_float)

    image1_float = image1.astype(np.float32)
    image2_is_scalar = np.isscalar(image2) if image2 is not None else False
    image2_float = None if image2 is None else (float(image2) if image2_is_scalar else image2.astype(np.float32))

    use_scaled_intensity_math = (
        not save_as_float
        and not image2_is_scalar
        and image2 is not None
        and np.issubdtype(dtype1, np.integer)
        and np.issubdtype(image2.dtype, np.integer)
    )
    dtype1_max = get_dtype_max(dtype1)

    if calculator_operation == 'add':
        result = image1_float + image2_float
    elif calculator_operation == 'subtract':
        result = image1_float - image2_float
    elif calculator_operation == 'multiply':
        if use_scaled_intensity_math:
            image2_max = get_dtype_max(image2.dtype)
            result = (image1_float / dtype1_max) * (image2_float / image2_max) * dtype1_max
        else:
            result = image1_float * image2_float
    elif calculator_operation == 'divide':
        result = np.zeros_like(image1_float, dtype=np.float32)
        if use_scaled_intensity_math:
            image2_max = get_dtype_max(image2.dtype)
            image1_norm = image1_float / dtype1_max
            image2_norm = image2_float / image2_max
            np.divide(image1_norm, image2_norm, out=result, where=image2_norm != 0)
            result *= dtype1_max
        else:
            np.divide(image1_float, image2_float, out=result, where=image2_float != 0)
    elif calculator_operation == 'min':
        result = np.minimum(image1_float, image2_float)
    elif calculator_operation == 'max':
        result = np.maximum(image1_float, image2_float)
    elif calculator_operation == 'average':
        result = (image1_float + image2_float) / 2.0
    else:
        raise ValueError(f"Unsupported calculator operation: {calculator_operation}")

    return cast_result(result, dtype1, save_as_float)


MANIFEST_PATH = sys.argv[1]
TASK_ID = int(sys.argv[2])
CALCULATOR_OPERATION = sys.argv[3]
SAVE_AS_FLOAT = sys.argv[4].lower() == 'true'

print("MANIFEST_PATH", MANIFEST_PATH)
print("TASK_ID", TASK_ID)
print("CALCULATOR_OPERATION", CALCULATOR_OPERATION)
print("SAVE_AS_FLOAT", SAVE_AS_FLOAT)

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

task_info = manifest[str(TASK_ID)]
input1_path = task_info['input1']
input2_path = task_info['input2']
input2_scalar = task_info['input2_scalar']
output_path = task_info['output']

print("input1_path", input1_path)
print("input2_path", input2_path)
print("input2_scalar", input2_scalar)
print("output_path", output_path)

image1 = tifffile.imread(input1_path)
if CALCULATOR_OPERATION == 'not':
    image2 = None
elif input2_scalar is not None:
    image2 = input2_scalar
else:
    image2 = tifffile.imread(input2_path)
result = apply_operation(image1, image2, CALCULATOR_OPERATION, SAVE_AS_FLOAT)
tifffile.imwrite(output_path, result)
print("Done")
