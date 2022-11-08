import json
import os
import time
from glob import glob
from pathlib import Path

from analysis import settings
from analysis.main import do_analysis


NODE_NAME = os.uname().nodename

json_settings = {}

def start_pipeline(settings_file_path):
    with open(settings_file_path, 'r') as f:
        settings_str = f.read()
        try:
            json_settings = json.loads(settings_str)
        except:
            log.warning("Unable to parse settings json")

    ROOT_DIR = json_settings.get('root_dir')
    IMS_FILE = json_settings.get('ims_file')
    if not ROOT_DIR and not IMS_FILE:
        raise RuntimeError("Root dir or imaris file required")
    elif not ROOT_DIR:
        ROOT_DIR = str(Path(IMS_FILE).parent)

    # SCAN = json_settings.get('scan', False)

    OUTPUT_FOLDER = json_settings.get('output_folder', os.path.join(ROOT_DIR, 'analysis'))
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    ACTIONS = json_settings.get('actions', ['register_brain'])

    # PRE_PROCESS = json_settings.get('pre_process', False)
    # PRE_PROCESSING_METHODS = json_settings.get('pre_processing_methods', [])
    # USE_DASK = json_settings.get('use_dask', False)


    analysis_dir_this_brain = os.path.join(OUTPUT_FOLDER, os.path.basename(IMS_FILE))
    if not os.path.exists(analysis_dir_this_brain):
        os.makedirs(analysis_dir_this_brain)
    do_analysis(IMS_FILE, analysis_dir_this_brain, ACTIONS)


while True:
    json_files = sorted(glob(os.path.join(settings.JSON_FOLDER, f'{NODE_NAME}_settings*.json')))
    if len(json_files):
        settings_file_path = json_files[0]
        setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
        start_pipeline(settings_file_path)
        os.remove(settings_file_path)
    else:
        time.sleep(30)
