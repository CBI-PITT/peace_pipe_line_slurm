import getpass
import os
from pathlib import Path


def init():
    global SETTINGS_FILE_PATH
    SETTINGS_FILE_PATH = ""


# JSON_FOLDERS = ['/h20/CBI/Iana/json', '/h20/Public/PEACE/JSON']
JSON_FOLDERS = ['/h20/CBI/Iana/json/test']
TRASH_FOLDER = "/h20/trash"


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')

USERNAME = getpass.getuser()
# USERNAME = 'lab'  # user that runs the pipeline
HOME = Path.home()

SLURM_JOBS_NICE_LEVEL = 50000000
SLURM_PARTITION_CPU = 'compute'
SLURM_PARTITION_GPU = 'gpu'
SLURM_PARTITION_HIGH_RAM = 'gpu'
SLURM_PARTITION_EXTREME = 'ai'

GPU_ENABLED_PARTITIONS = [
    SLURM_PARTITION_GPU,
    SLURM_PARTITION_EXTREME,
    ','.join([SLURM_PARTITION_EXTREME, SLURM_PARTITION_GPU]),
    ','.join([SLURM_PARTITION_GPU, SLURM_PARTITION_EXTREME])
]

PRIORITY_TO_NICE_MAP_GPU = {
    '0': 1000000,  # priority 1 for GPU;       priority 1 for compute n24 mem64
    '1': 129000,   # priority 18 for GPU;
    '2': 125000,   # priority 4018 for GPU;
    '3': 120000,   # priority 9018 for GPU;
    '4': 110000,   # priority 19018 for GPU;
    '5': 100000,    # priority 29018 for GPU;   priority 1 for compute n24 mem64
    '1313': 0
}

PRIORITY_TO_NICE_MAP_COMPUTE = {
    '0': 1000000,  # priority 1 for compute n24 mem64;      priority 1 for GPU
    '1': 61500,    # priority 77 for compute n24 mem64;
    '2': 61000,    # priority 577 for compute n24 mem64;
    '3': 60000,    # priority 1577 for compute n24 mem64;
    '4': 50000,    # priority 11577 for compute n24 mem64;
    '5': 40000,     # priority 21577 for compute n24 mem64;
    '1313': 0
}

TIFF_TILE_SIZE = (512, 512)

# UMASK = 0o002  # system mask for setting file and folder permissions (rw-rw-r--)
UMASK = 0o006  # system mask for setting file and folder permissions (rw-rw----)

ALLEN_RESOLUTION = 10
CELLFINDER_SOMA_DIAMETER = 10
CELLFINDER_THRESHOLD = 6
INFO_FILE_NAME = "dataset_info.json"
JOBS_FILE_NAME = "jobs.json"
ENABLE_JOB_HISTORY = True # _env_flag('PEACE_ENABLE_JOB_HISTORY', default=False)
USE_SACCT_FOR_HISTORY = True # _env_flag('PEACE_USE_SACCT_FOR_HISTORY', default=True)
JOB_HISTORY_DIR = os.environ.get('PEACE_JOB_HISTORY_DIR', os.path.join(JSON_FOLDERS[0], 'history'))
CELLFINDER_OUT_FOLDER_NAME = "output_full"
DEEPBLINK_OUT_FOLDER_NAME = "output_deepblink"
ATLAS_NAME_FORMAT = 'allen_mouse_{}um'
SUFFIX_100UM_VOLUME = '_100_100_100.tif'
DENOISE_FFT = True
PROCESSED_IMS_LOCATION = '/h20/CBI/Iana/processed_ims_files.json'
IN_PROGRESS_IMS_LOCATION = '/h20/CBI/Iana/in_progress_ims_files.json'
ATLAS_REFERENCE_PATH_FORMAT = '/h20/CBI/Iana/var/atlas_reference_{}um_wo_bckgnd.tif'  # atlas reference with background subtracted
DETECTED_SPOTS_FILE_NAME = 'all_detected_spots.npy'
CLASSIFIED_CELLS_FILE_NAME = 'classified_cells.npy'
DETECTED_SPOTS_IMARIS_FILE_NAME = 'all_detected_spots_imaris.npy'
TRANSFORMED_SPOTS_FILE_NAME = 'all_detected_spots_transformed.npy'
TRANSFORMED_CELLS_FILE_NAME = 'classified_cells_transformed.npy'
RESOLUTION_LEVEL_FOLDER_NAME = 'resolution_level_x'
RANDOM_FOREST_MASK_FOLDER_NAME = 'bg_fg_mask_apoc'
CELLS_DATAFRAME_NAME_PATTERN = 'spots_detailed_info_{}.csv'
LOG_FILE_NAME_PATTERN = "/h20/CBI/Iana/logs/analysis_pipeline_log_{}_{}.txt"
TIMESTAMP_FORAMT = '%Y-%m-%d_%H:%M:%S'
REGISTRATION_INFO_FILE_NAME = "registration_info.json"
MODEL_PATH_FILE = "model_to_use"  # model for cellfinder ResNet-50 classification
DEEPBLINK_MODEL_PATH = '/h20/CBI/Iana/src/deepblink/models/deepblink_particle.h5'  # model for deepBlink cell detection
DEEPBLINK_CHUNK_SIZE = (40, 1700, 3500)
DB_TYPE = "sqlite3"
MYSQL_DB_NAME = "temp_cells"
DB_LOCATION = ""

DASK_ALLOWED_NODES = ['pollux.cbiserver.pitt.edu.cbiserver.pitt.edu', 'deneb01', 'deneb02']  # machines allowed to use dask

PREPROCESSING_METHOD_PREFIX_MAP = {
    'stretch_contrast': 'contrast_stretched',
    'subtract_background': 'background_subtracted',
    'subtract_background_iterative': 'background_subtracted_iterative',
    'denoise_fft': 'denoised',
    'fft_1d_stripes_filter': '1d_fft_destriped',
    'fft_2d_notch_filter': 'notch_filtered'
}
PREPROCESSING_METHODS = [
    ['stretch_contrast'],
    ['fft_2d_notch_filter'],
    ['fft_2d_notch_filter', 'stretch_contrast'],
    # ['fft_2d_notch_filter', 'stretch_contrast', 'subtract_background_iterative']
]

# Example actions:
#   extract_tiff_series
#   pre_process,
#   register_brain,
#   get_best_registration,
#   detect_cells,
#   detect_cells_nn,
#   classify_cells,
#   save_cells_imaris,
#   analyze_cells_imaris,
#   analyze_cells_cellfinder,
#   visualize_cells
