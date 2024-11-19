import os
import subprocess
import time
from glob import glob

import numpy as np
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from imaris_ims_file_reader import ims


class brainreg:
    """
    Affine and nonlinear brain registration.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "brainreg",
        "extras": {
            "background_channel": 1,
            "atlas": "allen_mouse_25um",
            "orientation": "sal",
            "brain_geometry": "full"
        }
    }
    """
    def __init__(self, input, output, **kwargs):
        self.input = input
        self.output = output
        self.background_channel = kwargs.get('background_channel', 0)
        self.atlas = kwargs.get('atlas', "allen_mouse_25um")
        self.orientation = kwargs.get('orientation', "sal")
        self.brain_geometry = kwargs.get('brain_geometry', "full")

        self.output_operation_folder = os.path.join(self.output, 'brainreg')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.registration_folder = os.path.join(self.output_operation_folder, f"registration_{self.atlas}_channel_{self.background_channel}")
        self.ims_file = ims(self.input)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.registration_folder):
            os.makedirs(self.registration_folder)

    def run(self):
        print("Running brainreg")
        print("Input", self.input)
        print("Output", self.output)
        print("Channel", self.background_channel)
        self.calculate_resolution_level()    # calculate resolution level based on atlas
        self.extract_tiff_series()    # extract tiff series in SLURM
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.background_channel, 'shape')][-3]
        extracted_files = glob(os.path.join(self.output_operation_folder, f'resolution_level_{self.resolution_level}', f'channel_{self.background_channel}', "*.tif"))
        while len(extracted_files) < z_layers:
            extracted_files = glob(os.path.join(self.output_operation_folder, f'resolution_level_{self.resolution_level}', f'channel_{self.background_channel}', "*.tif"))
            print(f"Extracted files: {len(extracted_files)} of {z_layers}")
            time.sleep(10)
        self.run_registration()  # run brainreg in SLURM
        # delete tiff series  # TODO

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

    def extract_tiff_series(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.background_channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"extract_imaris_z_layers_rl{self.resolution_level}_c{self.background_channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "extract_imaris_z_layer.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(str(self.input))
            f.write(' ')
            f.write(str(self.output))
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.background_channel))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        #### run it on compute (cpu) partition
        command = ['sbatch', f'--array=0-{z_layers}', '-p', 'compute', '--mem=32Gb', '-n12', path_to_task]
        subprocess.run(command)

    def run_registration(self):
        path_to_task = os.path.join(self.jobs_folder, f"register_rl{self.resolution_level}_c{self.background_channel}_to_{self.atlas}.sh")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")  # TODO: create separate env for brainreg
            f.write('\n')
            f.write(f'brainreg {self.stack_to_register} {self.registration_folder}')
            f.write(f' -v {str(self.resolution[0])} {str(self.resolution[1])} {str(self.resolution[2])}')
            f.write(f' --orientation {self.orientation} --atlas {self.atlas} --brain_geometry {self.brain_geometry}')
            f.write('\n')

        print("Starting registration...")
        #### run on compute (cpu) partition
        command = ['sbatch', '-p', 'compute', '--mem=64Gb', '-n24', path_to_task]
        subprocess.run(command)

