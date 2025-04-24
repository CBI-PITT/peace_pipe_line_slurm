import json
import os
import subprocess
import time
from glob import glob

import numpy as np
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from imaris_ims_file_reader import ims

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class brainreg(ImageOperation):
    """
    Affine and nonlinear brain registration.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "brainreg",
        "extras": {
            "atlas": "allen_mouse_25um",
            "orientation": "sal",
            "brain_geometry": "full"
        }
    }
    """
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        if not self.output:
            self.output = self.metadata.get("base_output_dir", self.metadata.get("out_name"))
        self.background_channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.resolution = self.metadata['resolution']
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.atlas = kwargs.get('atlas', "allen_mouse_25um")
        self.orientation = kwargs.get('orientation', self.metadata['orientation'])
        self.brain_geometry = kwargs.get('brain_geometry', "full")

        self.output_operation_folder = os.path.join(self.output, 'brainreg')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        if self.metadata.get('sequence'):
            previous_operations = self.metadata['sequence'].split(',')
            previous_operation = f"_{previous_operations[-1]}" if len(previous_operations) > 1 else ""
        else:
            previous_operation = ""
        self.registration_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.background_channel}",
            f"registration_{self.atlas}{previous_operation}"
        )
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.registration_folder):
            os.makedirs(self.registration_folder)

    def run(self):
        print("Running brainreg")
        self.create_provenance()
        self.run_registration()  # run brainreg in SLURM

    def create_provenance(self):
        sequence = ",".join([self.metadata.get('sequence', ""), 'brainreg'])
        provenance = {
            "input": {
                "type": "tiff_series",  # input type
                "path": self.input,
            },
            "output": {
                "type": "folder",
                "path": self.registration_folder,
            },
            "process": {
                "parameters": {
                    "atlas": self.atlas,
                    "orientation": self.orientation,
                    "brain_geometry": self.brain_geometry
                }
            },
            "source": os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'),  # input provenance file
            "channel": self.background_channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": self.metadata.get('out_name', self.metadata['base_output_dir']),
            "base_input_dir": self.input,
            "sequence": sequence
        }
        with open(
                os.path.join(self.registration_folder, f'.{settings.INFO_FILE_NAME}'),
                "w") as f:
            f.write(json.dumps(provenance))

    def calculate_resolution_level(self):
        atlas = BrainGlobeAtlas(self.atlas)
        min_atlas_resolution = np.min(np.array(atlas.resolution))
        resolution_level = 0
        for rl in range(self.ims_file.ResolutionLevels-1, 0, -1):
            resolution = self.ims_file.metaData[(rl, 0, self.background_channel, 'resolution')][-3:]
            if np.all(np.round(np.array(resolution)) <= min_atlas_resolution):
                resolution_level = rl
                break
        self.resolution_level = resolution_level
        self.resolution = self.ims_file.metaData[(resolution_level, 0, self.background_channel, 'resolution')][-3:]
        print("Resolution level", resolution_level)
        print(self.ims_file.metaData[(resolution_level, 0, self.background_channel, 'resolution')])
        self.stack_to_register = os.path.join(self.output_operation_folder, f'resolution_level_{resolution_level}', f'channel_{self.background_channel}')
        if not os.path.exists(self.stack_to_register):
            os.makedirs(self.stack_to_register)

    def run_registration(self):
        path_to_task = os.path.join(self.jobs_folder, f"register_rl{self.resolution_level}_c{self.background_channel}_to_{self.atlas}.sh")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-brainreg")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate brainreg")
            f.write('\n')
            f.write('brainreg ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.registration_folder if ' ' not in self.registration_folder else f'"{self.registration_folder}"')
            f.write(f' -v {str(self.resolution[0])} {str(self.resolution[1])} {str(self.resolution[2])}')
            f.write(f' --orientation {self.orientation} --atlas {self.atlas} --brain_geometry {self.brain_geometry}')
            f.write('\n')

        print("Starting registration...")
        submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=24,
            memory=64,
            priority=self.priority
        )
