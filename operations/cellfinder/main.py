import json
import os
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from ..base import ImageOperation
from analysis import settings


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
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.signal_channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', 'lab')
        self.output_operation_folder = os.path.join(self.output, 'cellfinder')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.detection_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.signal_channel}",
            f"cellfinder_output"
        )
        self.resolution = self.metadata['resolution']
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)

    def run(self):
        print("Running cellfinder")
        # print("Input", self.input)
        # print("Output", self.output)
        # print("Channel", self.signal_channel)
        self.run_detection()  # run cellfinder in SLURM

    def run_detection(self):
        path_to_task = os.path.join(self.jobs_folder, f"cellfinder_rl{self.resolution_level}_c{self.signal_channel}.sh")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            # f.write('ulimit -n 600000')
            # f.write('\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-cellfinder")
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
        #### run on gpu partition
        command = [
            'sbatch',
            '-p', settings.SLURM_PARTITION_HIGH_RAM,
            '--mem=64Gb',
            '-n8',
            f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
            path_to_task
        ]
        subprocess.run(command)

