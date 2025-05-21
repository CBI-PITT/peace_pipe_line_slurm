"""
All PEACE operations work on folders with tif/tiff files.
These folders are the common medium that all readers should convert data to,
and all analysis plugins (operations) must take as an input.

Roadmap:

- tiff series reader (just provenance)
- description for each method
- description for each field
- partially processed folders - skip what's already done.
    Create multiple job arrays for continuous ranges of processed files
- instead of passing many command line arguments, pass the settings JSON file
- Add DBSCAN to deepblink? Or as a separate operation?
- use conda-pack to package all environments
"""


import importlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from glob import glob
from pathlib import Path

from analysis import settings
from analysis.main import do_analysis
from operations import *
from operations.base import ImageOperation, ImageReader


console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(
    settings.LOG_FILE_NAME_PATTERN.format(
        os.uname().nodename,
        datetime.now().strftime(settings.TIMESTAMP_FORAMT)
    )
)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler, file_handler]
)
log = logging.getLogger(__name__)


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
    # if not INPUT or not OUTPUT or not OPERATION:
    #     log.exception("Fields 'input', 'output' and 'operation' are required in the JSON")
    #     return
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


def start_workflow_slurm(settings_file_path):
    print("Starting workflow")
    with open(settings_file_path, 'r') as f:
        settings_str = f.read()
        try:
            json_settings = json.loads(settings_str)
        except:
            log.exception("Unable to parse settings json")
            return
    outputs = {}
    steps = json_settings["steps"]
    # operation_outputs = {x['operation']: x['output_name'] for x in steps}
    operation_outputs = {}

    for step in steps:
        operation_name = step['operation']
        print("Operation name", operation_name)
        extended_operation_name = operation_name
        if operation_name not in operation_outputs:
            operation_outputs[operation_name] = step['output_name']
        else:
            count = 1
            while extended_operation_name in operation_outputs:
                extended_operation_name = operation_name + str(count)
                count += 1
            operation_outputs[extended_operation_name] = step['output_name']
        try:
            operation_class = getattr(sys.modules[__name__], operation_name)
        except AttributeError:
            print(f"Attempting to load plugin for operation {operation_name}")
            operation_class = reader_plugins.get(operation_name)
            print('operation_class', operation_class)
            if not operation_class:
                operation_class = plugins.get(operation_name)

        input = ""
        output = ""
        extras = step["extras"].copy()
        extras.pop("operation")
        inputs_from_other_operations = step['input_bindings']
        print("inputs_from_other_operations", inputs_from_other_operations)
        print("operation_outputs", operation_outputs)
        for k, v in inputs_from_other_operations.items():
            previous_operation = [x for x in operation_outputs.keys() if operation_outputs[x] == v][0]
            print("previous_operation", previous_operation)
            previous_operation_info = outputs[previous_operation]
            previous_operation_provenance = json.load(open(previous_operation_info['provenance'], 'r'))
            print("previous_operation_provenance", previous_operation_provenance)
            previous_operation_output = previous_operation_provenance['output']['path']
            print("previous_operation_output", previous_operation_output)
            extras[k] = previous_operation_output
            if 'prerequisites' in extras:
                extras['prerequisites'].extend(previous_operation_info['prerequisites'])
            else:
                extras['prerequisites'] = previous_operation_info['prerequisites']
            if 'output' not in extras:
                extras["output"] = previous_operation_provenance["base_output_dir"]
        if 'input' in extras:
            input = extras.pop('input')
        if 'output' in extras:
            output = extras.pop('output')

        print("input", input)
        print("output", output)
        print("extras", extras)
        operation = operation_class(input, output, **extras)
        provenance, prerequisites = operation.run()
        print("================ got provenance:", provenance)
        print("================ got job ids:", prerequisites)
        outputs[extended_operation_name] = {'provenance': provenance, 'prerequisites': prerequisites}


while True:
    for json_folder in settings.JSON_FOLDERS:
        print(f"Looking for tasks in {json_folder}...")
        ### look for reader tasks ###
        reader_json_files = sorted(glob(os.path.join(json_folder, f'SLURM_reader*.json')))
        print(len(reader_json_files), "Reader JSON files found")
        for reader_json_file in reader_json_files:
            print("Starting processing")
            settings_file_path = reader_json_file
            setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
            try:
                start_reader_slurm(settings_file_path)
            except Exception as e:
                os.rename(settings_file_path, os.path.join(json_folder, 'err', os.path.basename(settings_file_path)))
                print(f"ERROR: {e}")
                print(traceback.format_exc())
            else:
                os.rename(settings_file_path, os.path.join(json_folder, 'done', os.path.basename(settings_file_path)))

        ### look for processing tasks ###
        json_files = sorted(glob(os.path.join(json_folder, f'SLURM_settings*.json')))
        print(len(json_files), "JSON files found")
        for json_file in json_files:
            print("Starting processing")
            settings_file_path = json_file
            setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
            try:
                start_pipeline_slurm(settings_file_path)
            except Exception as e:
                os.rename(settings_file_path,os.path.join(json_folder, 'err', os.path.basename(settings_file_path)))
                print(traceback.format_exc())
            else:
                os.rename(settings_file_path, os.path.join(json_folder, 'done', os.path.basename(settings_file_path)))

        ### look for workflows ###
        workflow_json_files = sorted(glob(os.path.join(json_folder, f'SLURM_workflow*.json')))
        print(len(workflow_json_files), "workflow JSON files found")
        for workflow_json_file in workflow_json_files:
            print("Starting processing")
            settings_file_path = workflow_json_file
            setattr(settings, "SETTINGS_FILE_PATH", settings_file_path)
            try:
                start_workflow_slurm(settings_file_path)
            except Exception as e:
                os.rename(settings_file_path, os.path.join(json_folder, 'err', os.path.basename(settings_file_path)))
                print(traceback.format_exc())
            else:
                os.rename(settings_file_path, os.path.join(json_folder, 'done', os.path.basename(settings_file_path)))

        print("Waining 30 seconds...")
        time.sleep(5)
