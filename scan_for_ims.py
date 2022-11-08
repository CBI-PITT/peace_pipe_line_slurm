from collections import defaultdict
from datetime import datetime
import glob
import json
import logging
import os
from pathlib import Path
import sys
import time

from analysis import settings
from analysis.main import do_analysis
from analysis.utils import read_in_progress_files_json, update_in_progress_files_json

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


NODE_NAME = os.uname().nodename


def scan(dict_of_dirs, ignore_processed=False):
    """
    Register all .ims files to the atlas in each of provided dirs.

    Search in the folders is done recursively.
    Files that have been found are checked against a list of processed .ims files.
    If they haven't been processed, they are sent to registration.
    :param dict_of_dirs: list of folders with data (.ims files)
    :param ignore_processed: do all requested processing regardles of these operations being done before
    :return: None
    """
    # Clear all in-progress lines corresponding to this machine
    update_in_progress_files_json(NODE_NAME, remove=True)

    for folder in dict_of_dirs.keys():
        log.info(f"Looking for .ims in {folder}...")
        # ims_file_paths = glob(folder + "/**/*.ims", recursive=True)

        operations_to_perform = dict_of_dirs[folder]  # functions to be computed for brains in this dir

        ims_file_paths = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.ims') and 'scene' not in file and not os.path.islink(os.path.join(root, file)):
                    ims_file_paths.append(os.path.join(root, file))

        log.info(f"Found {len(ims_file_paths)} .ims files")
        log.info(f"Preforming following operations: {operations_to_perform}")
        processed_ims_files_json = settings.PROCESSED_IMS_LOCATION
        in_progress_ims_files_json = settings.IN_PROGRESS_IMS_LOCATION
        processed_ims_files = {}
        analysis_dir = os.path.join(folder, 'analysis')
        if os.path.exists(processed_ims_files_json):
            with open(processed_ims_files_json, 'r') as f:
                input_data_str = f.read()
                processed_ims_files = json.loads(input_data_str)

        for ims_file_path in sorted(ims_file_paths):
            remaining_operations = operations_to_perform[:]  # TODO: read at every iteration
            if ims_file_path in processed_ims_files and not ignore_processed:
                # Some processing was previously done on this brain
                operations_done = processed_ims_files[ims_file_path]  # read from the file
                remaining_operations = list(set(operations_to_perform) - set(operations_done))
                if not remaining_operations:
                    # All processing has been done
                    log.warning(f"Skipping file because already processed: {ims_file_path}")
                    continue

            if os.path.exists(in_progress_ims_files_json):
                in_progress_ims_files = read_in_progress_files_json()
            if ims_file_path in in_progress_ims_files:
                # this file is being processed by another machine
                log.warning(f"File {ims_file_path} is being processed by other machine. Race condition possible.")
                # operations_being_done_by_other_machine = in_progress_ims_files[ims_file_path]
                # remaining_operations = list(set(remaining_operations) - set(operations_being_done_by_other_machine))
                log.warning("Skipping")
                continue

            analysis_dir_this_brain = os.path.join(analysis_dir, os.path.basename(ims_file_path))
            if not os.path.exists(analysis_dir_this_brain):
                os.makedirs(analysis_dir_this_brain)

            log.info(f"Working on {ims_file_path}")
            log.info(f"Saving results in {analysis_dir_this_brain}")
            do_analysis(ims_file_path, analysis_dir_this_brain, remaining_operations)
            log.info(f"Finished working on {ims_file_path}")


def do_priorities(tasks_by_node, ignore_processed=False):
    log.info("Doing priority tasks...")
    tasks = tasks_by_node[NODE_NAME]
    if os.path.exists(settings.PROCESSED_IMS_LOCATION):
        with open(settings.PROCESSED_IMS_LOCATION, 'r') as f:
            input_data_str = f.read()
            processed_ims_files = json.loads(input_data_str)
    for ims_file_path, operations_to_perform in tasks.items():
        if not ims_file_path or not operations_to_perform:
            continue
        remaining_operations = operations_to_perform[:]
        if ims_file_path in processed_ims_files and not ignore_processed:
            # Some processing was previously done on this brain
            operations_done = processed_ims_files[ims_file_path]  # read from the file
            remaining_operations = list(set(operations_to_perform) - set(operations_done))
        current_path = Path(ims_file_path)
        for level in range(len(current_path.parents) - 1):
            current_path = current_path.parent
            analysis_dir = str(current_path / 'analysis')
            if os.path.exists(analysis_dir):
                break
        analysis_dir_this_brain = os.path.join(analysis_dir, os.path.basename(ims_file_path))
        if not os.path.exists(analysis_dir_this_brain):
            os.makedirs(analysis_dir_this_brain)
        do_analysis(ims_file_path, analysis_dir_this_brain, remaining_operations)
    log.info("Done all priority tasks!")


if __name__ == "__main__":
    try:
        ignore_processed = False
        if len(sys.argv) > 1 and sys.argv[1] == 'ignore_processed':
            ignore_processed = True
        priorities = settings.PRIORITY_LIST
        do_priorities(priorities, ignore_processed=ignore_processed)

        folders_list = settings.ACTIONS[NODE_NAME]
        while True:
            scan(folders_list, ignore_processed=ignore_processed)
            wait_time_minutes = 30
            print(f"Waiting {wait_time_minutes} min ...")
            time.sleep(wait_time_minutes * 60)
    except KeyboardInterrupt:
        log.error("Interrupted")
        update_in_progress_files_json(NODE_NAME, remove=True)  # cleanup the json file
