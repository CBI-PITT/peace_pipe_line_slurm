import json
import os
import time

import numpy as np
import tifffile
import zarr

from operations.base import ImageReader
from analysis import settings
from analysis.guess_brain_orientation import _guess_orientation
from utils import get_user
from utils.slurm import submit_slurm_array, submit_slurm_job


class ome_zarr_reader(ImageReader):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'ome_zarr_reader'
        self.channel = int(kwargs.get('channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')

        self.jobs_folder = os.path.join(self.output, 'slurm_jobs')
        self.extracted_tiffs_folder = os.path.join(
            self.output,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}'
        )
        self.volume_100um_location = os.path.join(self.output, os.path.basename(self.input) + settings.SUFFIX_100UM_VOLUME)
        self.prerequisites = kwargs.get('prerequisites', [])
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.extracted_tiffs_folder):
            os.makedirs(self.extracted_tiffs_folder)

    def get_scale_path(self, resolution_level):
        return f'scale{resolution_level}'

    def get_scale_array(self, resolution_level):
        root = zarr.open(self.input, mode='r')
        return root[self.get_scale_path(resolution_level)]

    def load_metadata(self):
        attrs_path = os.path.join(self.input, '.zattrs')
        if os.path.exists(attrs_path):
            with open(attrs_path, 'r') as f:
                return json.load(f)
        return dict(zarr.open(self.input, mode='r').attrs)

    def run(self):
        print('Running OME Zarr Reader')
        metadata = self.initialize_info_file()
        with open(os.path.join(self.output, f'resolution_level_{self.resolution_level}', settings.INFO_FILE_NAME), 'w') as f:
            f.write(json.dumps(metadata))
        metadata['channel'] = self.channel
        with open(os.path.join(self.extracted_tiffs_folder, f'.{settings.INFO_FILE_NAME}'), 'w') as f:
            f.write(json.dumps(metadata))
        job_ids = self.extract_tiff_series()
        job_ids.extend(self.update_metadata())
        return os.path.join(self.extracted_tiffs_folder, f'.{settings.INFO_FILE_NAME}'), job_ids

    def extract_tiff_series(self):
        z_layers = self.get_scale_array(self.resolution_level).shape[-3]
        path_to_task = os.path.join(self.jobs_folder, f'extract_ome_zarr_z_layers_rl{self.resolution_level}_c{self.channel}.sh')
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), 'extract_z_layer.py')
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f'#SBATCH -J {self.user}-{self.name}')
            f.write('\n')
            f.write(f'#SBATCH -o {self.jobs_folder}/slurm_%j.out')
            f.write('\n')
            f.write('\n')
            f.write('source /h20/home/lab/miniconda3/bin/activate omehans-reader')
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(str(self.input if ' ' not in self.input else f'"{self.input}"'))
            f.write(' ')
            f.write(str(self.output if ' ' not in self.output else f'"{self.output}"'))
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'

        job_ids = submit_slurm_array(
            path_to_task,
            z_layers,
            partition=f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=12,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids

    def initialize_info_file(self):
        metadata = self.load_metadata()
        multiscale = metadata['multiscales'][0]
        scale_array = self.get_scale_array(self.resolution_level)
        full_resolution_array = self.get_scale_array(0)
        channel_metadata = metadata.get('omero', {}).get('channels', [])
        if channel_metadata:
            channel_maxima = [x.get('window', {}).get('max', 0) for x in channel_metadata]
            background_channel = int(np.argmin(channel_maxima))
            channels = int(len(channel_metadata))
        else:
            background_channel = 0
            channels = int(full_resolution_array.shape[1])

        options = {
            'out_name': self.output,
            'source': self.input,
            'channels': channels,
            'background_channel': background_channel,
            'volume_100um_location': self.volume_100um_location,
            'resolution': multiscale['datasets'][self.resolution_level]['coordinateTransformations'][0]['scale'][-3:],
            'shape': scale_array.shape[-3:],
            'resolution_level': int(self.resolution_level),
            'full_resolution': multiscale['datasets'][0]['coordinateTransformations'][0]['scale'],
            'full_shape': full_resolution_array.shape,
            'input': {
                'type': 'ome_zarr',
                'path': self.input
            },
            'output': {
                'type': 'tiff_series',
                'path': self.extracted_tiffs_folder
            },
            'process': {},
            'base_output_dir': self.output,
            'base_input_dir': '',
            'sequence': self.name
        }
        return options

    def update_metadata(self):
        path_to_task = os.path.join(
            self.jobs_folder,
            f'extract_ome_zarr_volume_100um_c{self.channel}.sh'
        )
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), 'extract_volume_at_resolution.py')
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f'#SBATCH -J {self.user}-ome-zarr-reader')
            f.write('\n')
            f.write(f'#SBATCH -o {self.jobs_folder}/slurm_%j.out')
            f.write('\n')
            f.write('\n')
            f.write('source /h20/home/lab/miniconda3/bin/activate omehans-reader')
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(str(self.input if ' ' not in self.input else f'"{self.input}"'))
            f.write(' ')
            f.write(str(self.volume_100um_location if ' ' not in self.volume_100um_location else f'"{self.volume_100um_location}"'))
            f.write(' ')
            f.write(str(self.channel))
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_CPU}',
            cores=4,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )

        while True:
            try:
                volume_100um = tifffile.imread(self.volume_100um_location)
                print('100um-volume has been extracted')
                break
            except:
                print('Waiting 10 seconds for 100um-volume to be extracted')
                time.sleep(10)

        orientation = _guess_orientation(volume_100um)
        metadata = self.initialize_info_file()
        metadata['orientation'] = orientation
        with open(os.path.join(self.output, f'resolution_level_{self.resolution_level}', settings.INFO_FILE_NAME), 'w') as f:
            f.write(json.dumps(metadata))
        metadata['channel'] = self.channel
        with open(os.path.join(self.extracted_tiffs_folder, f'.{settings.INFO_FILE_NAME}'), 'w') as f:
            f.write(json.dumps(metadata))

        return job_ids
