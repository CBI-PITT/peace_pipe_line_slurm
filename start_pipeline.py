"""
Roadmap:

- ResNet classification
- partially processed folders - skip what's already done.
    Create multiple job arrays for continuous ranges of processed files
"""


import importlib
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
from operations.base import ImageOperation, ImageReader


json_settings = {}


def load_plugins(plugin_folder):
    operations = {}
    for plugin_name in os.listdir(plugin_folder):
        plugin_path = os.path.join(plugin_folder, plugin_name)
        main_file = os.path.join(plugin_path, "main.py")

        if os.path.isdir(plugin_path) and os.path.isfile(main_file):
            # Dynamically load the main.py file
            spec = importlib.util.spec_from_file_location(f"{plugin_name}.main", main_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the class in the module that subclasses ImageOperation
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, ImageOperation) and obj is not ImageOperation:
                    operations[obj.__name__] = obj
    return operations


def load_reader_plugins(plugin_folder):
    operations = {}
    for plugin_name in os.listdir(plugin_folder):
        plugin_path = os.path.join(plugin_folder, plugin_name)
        main_file = os.path.join(plugin_path, "main.py")

        if os.path.isdir(plugin_path) and os.path.isfile(main_file):
            # Dynamically load the main.py file
            spec = importlib.util.spec_from_file_location(f"{plugin_name}.main", main_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the class in the module that subclasses ImageOperation
            for attr in dir(module):
                obj = getattr(module, attr)
                if isinstance(obj, type) and issubclass(obj, ImageReader) and obj is not ImageReader:
                    operations[obj.__name__] = obj
    return operations


plugin_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")
plugins = load_plugins(plugin_folder)
print("plugins", plugins)

reader_plugin_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reader_plugins")
reader_plugins = load_reader_plugins(reader_plugin_folder)
print("reader plugins", reader_plugins)


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
    try:
        operation_class = getattr(sys.modules[__name__], OPERATION)
    except AttributeError:
        print(f"Attempting to load plugin for operation {OPERATION}")
        operation_class = plugins[OPERATION]
    operation = operation_class(INPUT, OUTPUT, **EXTRAS)
    operation.run()


def start_reader_slurm(settings_file_path):
    print("Starting reader")

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
    try:
        operation_class = getattr(sys.modules[__name__], OPERATION)
    except AttributeError:
        print(f"Attempting to load plugin for operation {OPERATION}")
        operation_class = reader_plugins[OPERATION]
    operation = operation_class(INPUT, OUTPUT, **EXTRAS)
    operation.run()


while True:
    print("Looking for tasks...")
    ### look for reader tasks ###
    reader_json_files = sorted(glob(os.path.join(settings.JSON_FOLDER, f'SLURM_reader*.json')))
    print(len(reader_json_files), "Reader JSON files found")
    for reader_json_file in reader_json_files:
        print("Starting processing")
        settings_file_path = reader_json_file
        setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
        try:
            start_reader_slurm(settings_file_path)
        except Exception as e:
            os.rename(settings_file_path, os.path.join(settings.JSON_FOLDER, 'err', os.path.basename(settings_file_path)))
            print(f"ERROR: {e}")
            print(traceback.format_exc())
        else:
            os.rename(settings_file_path, os.path.join(settings.JSON_FOLDER, 'done', os.path.basename(settings_file_path)))

    ### look for processing tasks ###
    json_files = sorted(glob(os.path.join(settings.JSON_FOLDER, f'SLURM_settings*.json')))
    print(len(json_files), "JSON files found")
    for json_file in json_files:
        print("Starting processing")
        settings_file_path = json_file
        setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
        try:
            start_pipeline_slurm(settings_file_path)
        except Exception as e:
            os.rename(settings_file_path,os.path.join(settings.JSON_FOLDER, 'err', os.path.basename(settings_file_path)))
            print(f"ERROR: {e}")
            print(traceback.format_exc())
        else:
            os.rename(settings_file_path, os.path.join(settings.JSON_FOLDER, 'done', os.path.basename(settings_file_path)))

    print("Waining 30 seconds...")
    time.sleep(30)
