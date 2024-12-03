import os
import subprocess

import numpy as np
import bg_space as bgs
import tifffile
from imaris_ims_file_reader import ims
from skimage.transform import rescale
from bg_atlasapi.bg_atlas import BrainGlobeAtlas


class ants:
    """
    Affine and nonlinear brain registration.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "ants",
        "extras": {
            "background_channel": 0,
            "atlas": "allen_mouse_25um",
            "orientation": "sal",
        }
    }
    """
    def __init__(self, input, output, **kwargs):
        print("kwargs", kwargs)
        self.input = input
        self.output = output
        self.background_channel = int(kwargs.get('background_channel', 0))
        self.atlas = kwargs.get('atlas', "allen_mouse_25um")
        self.orientation = kwargs.get('orientation', "sal")

        self.output_operation_folder = os.path.join(self.output, 'ants')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.registration_folder = os.path.join(self.output_operation_folder, f"registration_{self.atlas}_channel_{self.background_channel}")
        self.ims_file = ims(self.input)
        self.stack_to_register = os.path.join(self.output, f"stack_c{self.background_channel}_rescaled_to_{self.atlas}.tif")
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.registration_folder):
            os.makedirs(self.registration_folder)

    def run(self):
        print("Running ants")
        print("Input", self.input)
        print("Output", self.output)
        print("Channel", self.background_channel)
        self.calculate_resolution_level()    # calculate resolution level based on atlas
        self.extract_atlas_resolution()    # extract multi-page tiff file from Imaris; downsample it to atlas resolution
        self.run_registration()  # run ants in SLURM
        # # delete tiff  # TODO

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

    def extract_atlas_resolution(self):
        atlas = BrainGlobeAtlas(self.atlas)
        raw = self.ims_file[self.resolution_level, 0, self.background_channel, :, :, :]
        print("RAW shape", raw.shape)
        print("Resolution", self.resolution)

        # TODO assuming that atlas is isotropic. Should be more general.
        raw_rescaled = rescale(raw, tuple(current / target for current, target in zip(self.resolution, atlas.resolution)))
        print("RAW rescaled shape", raw_rescaled.shape)

        raw_rescaled_reoriented = bgs.map_stack_to(self.orientation, atlas.orientation, raw_rescaled).astype('float32')
        print("RAW reoriented rescaled shape", raw_rescaled_reoriented.shape)

        self.stack_to_register = os.path.join(self.output, f"stack_c{self.background_channel}_rescaled_to_{self.atlas}.tif")
        tifffile.imwrite(self.stack_to_register, raw_rescaled_reoriented)

    def run_registration(self):
        path_to_task = os.path.join(self.jobs_folder, f"register_c{self.background_channel}_to_{self.atlas}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "register_ants.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate ants")
            f.write('\n')
            f.write(f'python {slurm_script} {self.stack_to_register} {self.registration_folder}')
            f.write(f' {self.atlas}')
            f.write('\n')

        print("Starting registration...")
        #### run on compute (cpu) partition
        command = ['sbatch', '-p', 'compute', '--mem=64Gb', '-n24', path_to_task]
        subprocess.run(command)
