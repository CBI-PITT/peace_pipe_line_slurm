import argparse
import json
import os
import sys
import uuid

import pandas as pd


def parse_key_value_pairs(pairs):
    kv_dict = {}
    for pair in pairs:
        if '=' not in pair:
            raise argparse.ArgumentTypeError(f"Invalid format for key-value pair: '{pair}'. Expected format: key=value")
        key, value = pair.split('=', 1)
        kv_dict[key] = value
    return kv_dict


parser = argparse.ArgumentParser()

# Three fixed arguments
parser.add_argument("cells_path")
parser.add_argument("results_folder")
parser.add_argument("options_path")

# Remaining key=value pairs (using nargs='*' to gather all remaining args)
parser.add_argument("pairs", nargs='*', help="Additional key=value arguments")

args = parser.parse_args()

# Convert key=value list to a dictionary
metadata_fields = parse_key_value_pairs(args.pairs)

cells_path = args.cells_path
results_folder = args.results_folder
options_path = args.options_path

print("cells_path", cells_path)
print("results_folder", results_folder)
print("options_path", options_path)

print("Metadata fields:", metadata_fields)

output_file_name = os.path.join(results_folder, f"{os.path.basename(cells_path).replace('.csv', '_with_metadata.csv')}")
if os.path.exists(output_file_name):
    print("Combined CSV file already exists")
    sys.exit(0)

# options = json.load(open(options_path, 'r'))
df = pd.read_csv(cells_path)
df_length = df.shape[0]

dataset_uuid = uuid.uuid4()
df['dataset_id'] = [dataset_uuid] * df_length

for k, v in metadata_fields.items():
    df[k] = [v] * df_length

df.to_csv(output_file_name, index=False)
print('DataFrame saved as', output_file_name)
