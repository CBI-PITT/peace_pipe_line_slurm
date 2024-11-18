import json
import os
import sys
import time
import traceback
from glob import glob
from pathlib import Path

from analysis import settings
from analysis.main import do_analysis
from operations import *


NODE_NAME = os.uname().nodename

json_settings = {}


def start_pipeline(settings_file_path):
    print("Starting pipeline")

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
    print("Actions before local settings:", ACTIONS)
    do_analysis(IMS_FILE, analysis_dir_this_brain, ACTIONS)


def start_pipeline_slurm(settings_file_path):
    print("Starting pipeline")

    with open(settings_file_path, 'r') as f:
        settings_str = f.read()
        try:
            json_settings = json.loads(settings_str)
        except:
            log.exception("Unable to parse settings json")
            return

    INPUT = json_settings.get('input')
    OUTPUT = json_settings.get('output')
    OPERATION = json_settings.get('operation')
    if not INPUT or not OUTPUT or not OPERATION:
        log.exception("Fields 'input', 'output' and 'operation' are required in the JSON")
        return
    EXTRAS = json_settings.get('extras', {})
    if type(EXTRAS) != dict:
        log.exception("Field 'extras' needs to be a mapping/dictionary")
        return
    operation_class = getattr(sys.modules[__name__], OPERATION)
    operation = operation_class(INPUT, OUTPUT, **EXTRAS)
    operation.run()


while True:
    print("Looking for tasks...")
    json_files = sorted(glob(os.path.join(settings.JSON_FOLDER, f'SLURM_settings*.json')))
    print(len(json_files), "JSON files found")
    if len(json_files):
        print("Starting processing")
        settings_file_path = json_files[0]
        setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
        try:
            start_pipeline_slurm(settings_file_path)
        except Exception as e:
            os.rename(settings_file_path,os.path.join(settings.JSON_FOLDER, 'err', os.path.basename(settings_file_path)))
            print(f"ERROR: {e}")
            print(traceback.format_exc())
        else:
            os.rename(settings_file_path, os.path.join(settings.JSON_FOLDER, 'done', os.path.basename(settings_file_path)))
    else:
        print("Waining 30 seconds...")
        time.sleep(30)
