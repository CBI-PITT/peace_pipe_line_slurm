import json
import os
import subprocess
from glob import glob

import numpy as np
import bg_space as bgs
import tifffile
from imaris_ims_file_reader import ims
from skimage.transform import rescale
from bg_atlasapi.bg_atlas import BrainGlobeAtlas

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class ants(ImageOperation):
    """
    Affine and nonlinear brain registration.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "ants",
        "extras": {
            "atlas": "allen_mouse_25um",
            "orientation": "sal",
        }
    }
    """
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.background_channel = int(self.metadata['channel'])
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.atlas = kwargs.get('atlas', "allen_mouse_25um")
        self.orientation = kwargs.get('orientation', "sal")
        self.resolution_level = int(self.metadata['resolution_level'])
        self.resolution = self.metadata['resolution']

        self.output_operation_folder = os.path.join(self.output, 'ants')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.registration_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.background_channel}",
            f"registration_{self.atlas}"
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.registration_folder):
            os.makedirs(self.registration_folder)

    def run(self):
        print("Running ants")
        # print("Input", self.input)
        # print("Output", self.output)
        # print("Channel", self.background_channel)
        self.run_registration()  # run ants in SLURM

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
        if self.metadata["source"].endswith('.ims'):
            from imaris_ims_file_reader import ims
            ims_file = ims(self.metadata["source"])
            raw = ims_file[self.resolution_level, 0, self.background_channel, :, :, :]  # TODO: extract the atlas resolution directly
        else:
            def read_image(file_path):
                return tifffile.imread(file_path)

            image_files = sorted(glob(os.path.join(self.input, '*.tif*')))
            z, y, x = self.metadata['shape']
            lazy_arrays = [
                da.from_delayed(da.delayed(read_image)(f), shape=(y, x), dtype='uint16')
                for f in image_files
            ]
            dask_array = da.stack(lazy_arrays, axis=0)  # Shape: (z, y, x)
            raw = dask_array.compute()  # TODO rescale individual z slices first
        print("RAW shape", raw.shape)

        # TODO assuming that atlas is isotropic. Should be more general.
        raw_rescaled = rescale(raw, tuple(current / target for current, target in zip(self.resolution, atlas.resolution)))
        print("RAW rescaled shape", raw_rescaled.shape)

        raw_rescaled_reoriented = bgs.map_stack_to(self.orientation, atlas.orientation, raw_rescaled).astype('float32')
        print("RAW reoriented rescaled shape", raw_rescaled_reoriented.shape)

        tifffile.imwrite(self.stack_to_register, raw_rescaled_reoriented)

    def run_registration(self):
        path_to_task = os.path.join(self.jobs_folder, f"register_c{self.background_channel}_to_{self.atlas}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "register_ants.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-ants")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate ants")
            f.write('\n')
            f.write(f'python {slurm_script} ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.registration_folder if ' ' not in self.registration_folder else f'"{self.registration_folder}"')
            f.write(f' {self.atlas}')
            f.write(f' {self.orientation}')
            f.write('\n')

        print("Starting registration...")
        submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=24,
            memory=64,
            priority=self.priority
        )
        # command = [
        #     'sbatch',
        #     '-p', f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
        #     '--mem=64Gb',
        #     '-n24',
        #     f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
        #     path_to_task
        # ]
        # subprocess.run(command)
