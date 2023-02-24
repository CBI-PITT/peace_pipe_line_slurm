def init():
    global SETTINGS_FILE_PATH
    SETTINGS_FILE_PATH = ""


JSON_FOLDER = '/CBI_Hive/CBI/Iana/json'

ALLEN_RESOLUTION = 10
CELLFINDER_SOMA_DIAMETER = 10
CELLFINDER_THRESHOLD = 6
INFO_FILE_NAME = "dataset_info.json"
CELLFINDER_OUT_FOLDER_NAME = "output_full"
DEEPBLINK_OUT_FOLDER_NAME = "output_deepblink"
ATLAS_NAME_FORMAT = 'allen_mouse_{}um'
SUFFIX_100UM_VOLUME = '_100_100_100.tif'
DENOISE_FFT = True
PROCESSED_IMS_LOCATION = '/CBI_Hive/CBI/Iana/processed_ims_files.json'
IN_PROGRESS_IMS_LOCATION = '/CBI_Hive/CBI/Iana/in_progress_ims_files.json'
ATLAS_REFERENCE_PATH_FORMAT = '/CBI_Hive/CBI/Iana/var/atlas_reference_{}um_wo_bckgnd.tif'  # atlas reference with background subtracted
DETECTED_SPOTS_FILE_NAME = 'all_detected_spots.npy'
CLASSIFIED_CELLS_FILE_NAME = 'classified_cells.npy'
DETECTED_SPOTS_IMARIS_FILE_NAME = 'all_detected_spots_imaris.npy'
TRANSFORMED_SPOTS_FILE_NAME = 'all_detected_spots_transformed.npy'
TRANSFORMED_CELLS_FILE_NAME = 'classified_cells_transformed.npy'
RESOLUTION_LEVEL_FOLDER_NAME = 'resolution_level_x'
RANDOM_FOREST_MASK_FOLDER_NAME = 'bg_fg_mask_apoc'
CELLS_DATAFRAME_NAME_PATTERN = 'spots_detailed_info_{}.csv'
LOG_FILE_NAME_PATTERN = "/CBI_Hive/CBI/Iana/logs/analysis_pipeline_log_{}_{}.txt"
TIMESTAMP_FORAMT = '%Y-%m-%d_%H:%M:%S'
REGISTRATION_INFO_FILE_NAME = "registration_info.json"
MODEL_PATH_FILE = "model_to_use"  # model for cellfinder ResNet-50 classification
DEEPBLINK_MODEL_PATH = '/CBI_Hive/CBI/Iana/src/deepblink/models/deepblink_particle.h5'  # model for deepBlink cell detection
DEEPBLINK_CHUNK_SIZE = (40, 1700, 3500)

DASK_ALLOWED_NODES = ['pollux', 'deneb']  # machines allowed to use dask

PREPROCESSING_METHOD_PREFIX_MAP = {
    'stretch_contrast': 'contrast_stretched',
    'subtract_background': 'background_subtracted',
    'subtract_background_iterative': 'background_subtracted_iterative',
    'denoise_fft': 'denoised',
    'fft_1d_stripes_filter': '1d_fft_destriped',
    'fft_2d_notch_filter': 'notch_filtered'
}
PREPROCESSING_METHODS = [
    ['fft_2d_notch_filter'],
    ['fft_2d_notch_filter', 'stretch_contrast'],
    ['fft_2d_notch_filter', 'stretch_contrast', 'subtract_background_iterative']
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

