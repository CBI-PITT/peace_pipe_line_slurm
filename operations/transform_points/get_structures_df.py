import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import bg_space as bgs
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from cellfinder.analyse.analyse import transform_points_to_downsampled_space
from cellfinder.main import get_downsampled_space
from imlib.IO.cells import get_cells

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)


def transform_points_downsampled_to_atlas_space(
        downsampled_points, atlas, deformation_field_paths, output_filename=None
):
    field_scales = [int(1000 / resolution) for resolution in atlas.resolution]
    points = [[], [], []]
    for axis, deformation_field_path in enumerate(deformation_field_paths):
        deformation_field = tifffile.imread(deformation_field_path)
        for point in downsampled_points:
            try:
                point = [int(round(p)) for p in point]
                points[axis].append(
                    int(
                        round(
                            field_scales[axis]
                            * deformation_field[point[0], point[1], point[2]]
                        )
                    )
                )
            except IndexError:
                print(
                    f'IndexError when transforming point ({point[0]},{point[1]},{point[2]}) from downsampled to atlas space.'
                )
    transformed_points = np.array(points).T

    if output_filename is not None:
        df = pd.DataFrame(transformed_points)
        df.to_hdf(output_filename, key="df", mode="w")

    return transformed_points


cells_path = sys.argv[1]
registration_path = sys.argv[2]
results_folder = sys.argv[3]
metadata = sys.argv[4]

options = json.load(open(metadata, 'r'))

deformation_field_paths = [
    os.path.join(registration_path, 'deformation_field_0.tiff'),
    os.path.join(registration_path, 'deformation_field_1.tiff'),
    os.path.join(registration_path, 'deformation_field_2.tiff')
]

# atlas = BrainGlobeAtlas('allen_mouse_{}um'.format(options['allen_resolution']))  # TODO

registration_json = os.path.join(registration_path, 'brainreg.json')
registration_data = json.load(open(registration_json, 'r'))
atlas = BrainGlobeAtlas(registration_data['atlas'])

source_space = bgs.AnatomicalSpace(
    registration_data['orientation'],
    shape=options['shape'],
    resolution=options['resolution'],
)

downsampled_space = get_downsampled_space(
    atlas,
    os.path.join(registration_path, 'boundaries.tiff')
)

classification_df = pd.read_csv(cells_path)
classification_df_coords = classification_df[["axis-0", "axis-1", "axis-2"]]
all_detected_spots = classification_df_coords.to_numpy()
all_detected_spots_downsampled = transform_points_to_downsampled_space(
    all_detected_spots, downsampled_space, source_space
)
all_detected_spots_transformed = transform_points_downsampled_to_atlas_space(
    all_detected_spots_downsampled, atlas, deformation_field_paths
)

all_detected_spots_downsampled = np.round(all_detected_spots_downsampled).astype(int)
# For each point, get atlas label
label_ids = []
empty_points = []
error_points = []
good_points = []
df_data = []

signal_channels = [1]
signal_channel = signal_channels[0]

for ind in range(all_detected_spots_transformed.shape[0]):
    label_id = 0
    structure_code = ''
    structure_name = ''

    if np.any(all_detected_spots_transformed[ind] < 0):
        error_points.append([
            all_detected_spots_transformed[ind, 0],
            all_detected_spots_transformed[ind, 1],
            all_detected_spots_transformed[ind, 2]
        ])
        print("Point with negative coordinates")
        continue

    try:  # some points are outside atlas
        atlas_value = atlas.annotation[
            all_detected_spots_transformed[ind, 0],
            all_detected_spots_transformed[ind, 1],
            all_detected_spots_transformed[ind, 2]
        ]
    except IndexError as e:
        error_points.append([
            all_detected_spots_transformed[ind, 0],
            all_detected_spots_transformed[ind, 1],
            all_detected_spots_transformed[ind, 2]
        ])
    else:  # if we didn't get exception
        df_row = atlas.lookup_df.index[atlas.lookup_df['id'] == atlas_value]
        if not df_row.empty:  # such atlas_value exists
            row_values = atlas.lookup_df.iloc[df_row]
            structure_name = row_values['name'].values[0]
            structure_code = row_values['acronym'].values[0]
            label_id = atlas_value
            good_points.append([
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ])
        else:  # no such atlas_value (it's usually 0 in this case)
            empty_points.append([
                all_detected_spots_transformed[ind, 0],
                all_detected_spots_transformed[ind, 1],
                all_detected_spots_transformed[ind, 2]
            ])
    finally:  # it will run either way
        if any([x < 0 for x in all_detected_spots_transformed[ind, :]]):
            label_id = 0
            structure_code = ''
            structure_name = ''

        label_ids.append(label_id)
        data_entry = [
            uuid.uuid4(),
            0,  # time_point
            signal_channel,  # channel
            all_detected_spots[ind, 0] * options['resolution'][0],
            # 'z_raw'  # TODO take from original imaris points?
            all_detected_spots[ind, 1] * options['resolution'][1],  # 'y_raw',
            all_detected_spots[ind, 2] * options['resolution'][2],  # 'x_raw',
            'um',  # 'raw_coord_units'
            int(round(all_detected_spots[ind, 0] * options['resolution'][0] / options['full_resolution'][0])),
            # 'z_raw_px'  # TODO take from original imaris points?
            int(round(all_detected_spots[ind, 1] * options['resolution'][1] / options['full_resolution'][1])),
            # 'y_raw_px'
            int(round(all_detected_spots[ind, 2] * options['resolution'][2] / options['full_resolution'][2])),
            # 'x_raw_px'
            1,  # 'is_cell',
            '',  # 'type',
            atlas.atlas_name,  # 'atlas_name',
            # options["allen_resolution"],  # 'atlas_resolution',
            all_detected_spots_downsampled[ind, 0],  # 'z_downsampled',
            all_detected_spots_downsampled[ind, 1],  # 'y_downsampled',
            all_detected_spots_downsampled[ind, 2],  # 'x_downsampled',
            all_detected_spots_transformed[ind, 0] * atlas.resolution[0],  # 'z_transformed',
            all_detected_spots_transformed[ind, 1] * atlas.resolution[1],  # 'y_transformed',
            all_detected_spots_transformed[ind, 2] * atlas.resolution[2],  # 'x_transformed',
            'um',  # transformed_coord_units
            int(round(all_detected_spots_transformed[ind, 0])),  # 'z_transformed_px',
            int(round(all_detected_spots_transformed[ind, 1])),  # 'y_transformed_px',
            int(round(all_detected_spots_transformed[ind, 2])),  # 'x_transformed_px',
            structure_name,  # 'atlas_structure_name',
            structure_code,  # 'atlas_structure_acronym',
            label_id,  # 'atlas_structure_number',
            # metadata_id,  # 'metadata'
        ]
        df_data.append(data_entry)

# create a DataFrame
df_column_names = [
    'uuid',
    'time_point',
    'channel',
    'z_raw',
    'y_raw',
    'x_raw',
    'raw_coord_units',
    'z_raw_px',
    'y_raw_px',
    'x_raw_px',
    'is_cell',
    'type',
    'atlas_name',
    # 'atlas_resolution',
    'z_downsampled',
    'y_downsampled',
    'x_downsampled',
    'z_transformed',
    'y_transformed',
    'x_transformed',
    'transformed_coord_units',
    'z_transformed_px',
    'y_transformed_px',
    'x_transformed_px',
    'atlas_structure_name',
    'atlas_structure_acronym',
    'atlas_structure_number',
    # 'metadata'
]

df = pd.DataFrame(df_data, columns=df_column_names)
output_csv_file_path = os.path.join(
    results_folder,
    f"{os.path.basename(cells_path.replace('.csv', ''))}_for_dashboard.csv"
)

df = df.astype(
    {"uuid": str, "z_raw": float, "y_raw": float, "x_raw": float, "raw_coord_units": str, "z_raw_px": int,
     "y_raw_px": int, "x_raw_px": int, "is_cell": int, "type": str, "atlas_name": str, # "atlas_resolution": str,
     "z_downsampled": int, "y_downsampled": int, "x_downsampled": int, "z_transformed": float,
     "y_transformed": float, "x_transformed": float, "transformed_coord_units": str, "z_transformed_px": int,
     "y_transformed_px": int, "x_transformed_px": int, "atlas_structure_name": str,
     "atlas_structure_acronym": str,
     "atlas_structure_number": int} #, "metadata": int}
)
df.to_csv(output_csv_file_path, index=False)
print('DataFrame saved as', output_csv_file_path)
