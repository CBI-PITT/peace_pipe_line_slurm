import json
import os
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class cellfinder(ImageOperation):
    """
    Thresholding-based cell detection.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "cellfinder",
        "extras": {}
    }
    """
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "cellfinder"
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.signal_channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.output_operation_folder = os.path.join(self.output, self.name)
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        if self.metadata.get('sequence'):
            previous_operations = self.metadata['sequence'].split(',')
            previous_operation = f"_{previous_operations[-1]}" if len(previous_operations) > 1 else ""
        else:
            previous_operation = ""
        self.detection_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.signal_channel}",
            f"cellfinder_output{previous_operation}"
        )
        self.out_csv_path = os.path.join(self.detection_folder, "points", "cells.xml")
        self.resolution = self.metadata['resolution']
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Cellfinder prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)

    def run(self):
        print("Running cellfinder")
        provenance_file_path = self.create_provenance()
        job_ids = []
        if not os.path.exists(self.out_csv_path):
            job_ids = self.run_detection()  # run cellfinder in SLURM
        else:
            print("Output file already exists")
        return provenance_file_path, job_ids

    def create_provenance(self):
        sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        provenance = {
            "input": {
                "type": "tiff_series",  # input type
                "path": self.input,
            },
            "output": {
                "type": "csv",
                "path": self.out_csv_path,
            },
            "process": {
                "parameters": {
                }
            },
            "source": os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'),  # input provenance file
            "channel": self.signal_channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": self.metadata.get('out_name', self.metadata["base_output_dir"]),
            "base_input_dir": self.input,
            "sequence": sequence
        }
        provenance_file_path = os.path.join(self.detection_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def run_detection(self):
        path_to_task = os.path.join(self.jobs_folder, f"cellfinder_rl{self.resolution_level}_c{self.signal_channel}.sh")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            # f.write('ulimit -n 600000')
            # f.write('\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-{self.name}")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate cellfinder")
            f.write('\n')
            f.write('cellfinder -s ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' -b ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' -o ')
            f.write(self.detection_folder if ' ' not in self.detection_folder else f'"{self.detection_folder}"')
            f.write(f' -v {str(self.resolution[0])} {str(self.resolution[1])} {str(self.resolution[2])}')
            f.write(f' --orientation sal --atlas allen_mouse_25um')
            f.write(f' --no-analyse --no-figures --no-register --no-classification --ball-z-size {str(int(self.resolution[0])+1)}')
            f.write('\n')

        print("Starting cellfinder detection...")
        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=8,
            memory=64,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
