import os
import sys
from pathlib import Path

from imaris_ims_file_reader import ims
import tifffile

this_script = Path(__file__)
parent_folder = this_script.parent
operations_folder = parent_folder.parent
project_root = operations_folder.parent
sys.path.append(str(project_root))

from analysis import settings

os.umask(settings.UMASK)


IMS_FILE_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
resolution_level = sys.argv[3]
channel = sys.argv[4]
z = sys.argv[5]
# compress = int(sys.argv[6])
y_start = int(sys.argv[6])
y_end = int(sys.argv[7])
x_start = int(sys.argv[8])
x_end = int(sys.argv[9])

print("IMS_FILE_PATH", IMS_FILE_PATH)
print("OUTPUT_DIR", OUTPUT_DIR)
print("resolution_level", resolution_level)
print("channel", channel)
print("z", z)

# full_output_folder = os.path.join(OUTPUT_DIR, f'resolution_level_{resolution_level}', f'channel_{channel}')
full_output_path = os.path.join(
    OUTPUT_DIR,
    f"r{str(resolution_level).zfill(2)}_t00_c{str(channel).zfill(2)}_z{str(z).zfill(4)}.tif"
)
if os.path.exists(full_output_path):
    print(f"File for z={z} already exists")
    sys.exit(0)

f = ims(IMS_FILE_PATH)
plane = f[int(resolution_level), 0, int(channel), int(z), y_start:y_end, x_start:x_end]
try:
    os.makedirs(full_output_folder)
except:
    pass

# if compress:
#     tifffile.imwrite(full_output_path, plane, tile=(512, 512), compression="zlib")
# else:
tifffile.imwrite(full_output_path, plane)
