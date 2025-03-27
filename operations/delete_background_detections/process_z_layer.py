import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile


INFO_FILE_NAME = "dataset_info.json"


def remove_background_detections(options):
    """
    Remove false positive cell detections in the background.

    """
    print("Removing background detections...")

    def process(df_part):
        points = df_part[['axis-1', 'axis-2']].to_numpy()
        detected_cells_np = np.floor(points).astype(int)
        # create a binary image where points are represented as pixels with value 1, the rest of the image is 0
        cells_binary = np.zeros(options['shape'][1:], dtype=np.uint8)
        np.put(cells_binary, np.ravel_multi_index(detected_cells_np.T, options['shape'][1:]), 1)
        # read the mask
        # multiply cells image by mask
        cells_filtered = cells_binary * mask
        # get indices of points that get removed, "unravel" them
        # convert binary image back to points
        nz = np.nonzero(cells_filtered)
        zipped_nz = list(zip(*nz))
        filtered_cells_np = np.asarray(zipped_nz)
        # save points as dataframe
        filtered_cells_df = pd.DataFrame()
        if filtered_cells_np.shape[0]:
            filtered_cells_df['axis-0'] = [z_layer] * filtered_cells_np.shape[0]
            filtered_cells_df['axis-1'] = list(filtered_cells_np[:, 0])
            filtered_cells_df['axis-2'] = list(filtered_cells_np[:, 1])
        return filtered_cells_df

    try:
        df = pd.read_csv(POINTS_DF)
        # print("DF", df.shape)
        df = df.round().astype('int')
        # print(df.head())
        partial_df = df[df["axis-0"] == z_layer]
        print("partial DF", partial_df.shape)

        mask = tifffile.imread(
            os.path.join(MASKS_DIR, f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z_layer).zfill(4)}.tif")  # TODO naming can differ bw methods
        )
        mask = (mask >= 0.5).astype('uint8')
        print("min mask", mask.min())
        print("max mask", mask.max())

        filtered_cells_df = pd.DataFrame()
        if partial_df.shape[0]:
            filtered_cells_df = process(partial_df)

        print("Saving df to csv")
        filtered_cells_df.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f'partial_df_z{str(z_layer).zfill(5)}.csv'
            )
        )
    except:
        df = pd.DataFrame()
        df.to_csv(os.path.join(OUTPUT_DIR, f'partial_df_z{str(z_layer).zfill(5)}.csv'))


INPUT_TIFF_STACK_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
POINTS_DF = sys.argv[3]
MASKS_DIR = sys.argv[4]
z_layer = int(sys.argv[5])

# print('INPUT_TIFF_STACK_DIR', INPUT_TIFF_STACK_DIR)
# print('OUTPUT_DIR', OUTPUT_DIR)
# print('POINTS_DF', POINTS_DF)
# print('MASKS_DIR', MASKS_DIR)
# print('z_layer', z_layer)

metadata = json.load(open(os.path.join(INPUT_TIFF_STACK_DIR, f'.{INFO_FILE_NAME}'), 'r'))
resolution_level = metadata['resolution_level']
channel = metadata['channel']

# print('resolution_level', resolution_level)
# print('channel', channel)

remove_background_detections(metadata)
