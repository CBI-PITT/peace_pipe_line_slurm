import getpass
from pathlib import Path


def init():
    global SETTINGS_FILE_PATH
    SETTINGS_FILE_PATH = ""


JSON_FOLDERS = [Path.home() / "json"]
TRASH_FOLDER = Path.home() / "trash"

USERNAME = getpass.getuser()
# USERNAME = 'lab'  # user that runs the pipeline
HOME = Path.home()
CONTAINER_RUNTIME = 'apptainer'
CONTAINER_FALLBACK_DIR = Path.home() / "containers"

SLURM_JOBS_NICE_LEVEL = 50000000
SLURM_PARTITION_CPU = 'compute'
SLURM_PARTITION_GPU = 'compute'
SLURM_PARTITION_HIGH_RAM = 'compute'
SLURM_PARTITION_EXTREME = 'compute'

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

# ALLEN_RESOLUTION = 10
INFO_FILE_NAME = "dataset_info.json"
JOBS_FILE_NAME = "jobs.json"
SUFFIX_100UM_VOLUME = '_100_100_100.tif'
# PROCESSED_IMS_LOCATION = '/h20/CBI/Iana/processed_ims_files.json'
# IN_PROGRESS_IMS_LOCATION = '/h20/CBI/Iana/in_progress_ims_files.json'
LOG_FILE_NAME_PATTERN = str(Path.home() / "logs" / "analysis_pipeline_log_{}_{}.txt")
TIMESTAMP_FORAMT = '%Y-%m-%d_%H:%M:%S'
# REGISTRATION_INFO_FILE_NAME = "registration_info.json"
DEEPBLINK_MODEL_PATH = Path.home() / "models" / "deepblink_particle.h5"  # model for deepBlink cell detection
DEEPBLINK_CHUNK_SIZE = (40, 1700, 3500)
# DB_TYPE = "sqlite3"
# MYSQL_DB_NAME = "temp_cells"
# DB_LOCATION = ""
