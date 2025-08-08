import json
import os

import dask.array as da
import numpy as np
import tifffile
from imaris_ims_file_reader import ims

from analysis import settings


def create_chunk_json(number, chunk, json_path):
    chunk_meta = {
        "shape": list(chunk.shape),
        "min": int(chunk.min()),
        "max": int(chunk.max()),
    }
    if number == 0:
        chunk_meta["content"] = "bg"
        chunk_meta["thr"] = round(float((chunk.mean() + 10. * chunk.std())), 2)
    else:
        chunk0_json = os.path.join(os.path.dirname(json_path), ".chunk_00000.json")
        chunk0_meta = json.load(open(chunk0_json, 'r'))
        total_voxels = chunk.shape[0] * chunk.shape[1] * chunk.shape[2]
        binary = chunk > float(chunk0_meta["thr"])
        outliers = np.count_nonzero(binary)
        outliers_percent = float(outliers) / total_voxels * 100
        print("Outliers: ", outliers_percent, "%")
        if outliers_percent < 0.01:
            chunk_meta["content"] = "bg"
        else:
            chunk_meta["content"] = "fg"
    print("chunk_meta", chunk_meta)
    json.dump(chunk_meta, open(json_path, "w"))


def extract_chunk_from_imaris_by_number(number, chunks_folder, ims_file_path, resolution_level, channel):
    print("Extracting chunks from the imaris file")
    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    chunk_json = os.path.join(chunks_folder, f".chunk_{str(number).zfill(5)}.json")
    if os.path.exists(chunk_file) and os.path.exists(chunk_json):
        return
    chunk_indices_path = os.path.join(chunks_folder, 'chunk_indices.npy')
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)
    slices = chunk_indices[number]
    ims_file = ims(ims_file_path, ResolutionLevelLock=resolution_level)
    ims_file_dask = da.array(ims_file)
    tiffstack = ims_file_dask[0, channel, :, :, :]
    chunk = tiffstack[tuple(slices)]
    print(chunk.shape)
    chunk = chunk.compute()
    if not os.path.exists(chunk_file):
        tifffile.imwrite(chunk_file, chunk)
    if not os.path.exists(chunk_json):
        create_chunk_json(number, chunk, chunk_json)


def extract_chunk_from_tiff_series_by_number(number, chunks_folder, input_dir):
    print("Extracting chunks from the tiff series")

    from dask import array as da
    from dask import delayed

    chunk_file = os.path.join(chunks_folder, f"chunk_{str(number).zfill(5)}.tif")
    chunk_json = os.path.join(chunks_folder, f".chunk_{str(number).zfill(5)}.json")
    if os.path.exists(chunk_file) and os.path.exists(chunk_json):
        return

    # Get a sorted list of all image file paths
    image_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tif', '.tiff'))]
    )
    metadata = json.load(open(os.path.join(input_dir, f'.{settings.INFO_FILE_NAME}'), 'r'))
    z, y, x = metadata['shape']

    # Function to load a single image
    @delayed
    def read_image(file_path):
        return tifffile.imread(file_path)

    # Create a Dask array
    lazy_arrays = [da.from_delayed(read_image(f), shape=(y, x), dtype='uint16') for f in image_files]
    stacked_array = da.stack(lazy_arrays, axis=0)  # Stack along a new axis to get shape (z, y, x)

    # Check the resulting array shape and type
    print(stacked_array.shape)

    chunk_indices_path = os.path.join(chunks_folder, 'chunk_indices.npy')
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)
    slices = chunk_indices[number]
    chunk = stacked_array[tuple(slices)]
    print(chunk.shape)
    chunk = chunk.compute()
    if not os.path.exists(chunk_file):
        tifffile.imwrite(chunk_file, chunk)
    if not os.path.exists(chunk_json):
        create_chunk_json(number, chunk, chunk_json)
