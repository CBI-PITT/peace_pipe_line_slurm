import json
import os
import re
import shlex
import subprocess
from glob import glob

from analysis import settings
from operations.base import ImageReader
from utils import get_user
from utils.slurm import split_slurm_array, submit_slurm_array


class jp2_reader(ImageReader):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'jp2_reader'
        self.channel = 0
        self.resolution_level = 0
        self.resolution_z = float(kwargs.get('resolution_z', 1))
        self.resolution_y = float(kwargs.get('resolution_y', 1))
        self.resolution_x = float(kwargs.get('resolution_x', 1))
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.prerequisites = kwargs.get('prerequisites', [])

        self.jobs_folder = os.path.join(self.output, 'slurm_jobs')
        self.extracted_tiffs_folder = os.path.join(
            self.output,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}'
        )
        self.jp2_files = self.get_jp2_files()
        self._shape = None

        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.extracted_tiffs_folder):
            os.makedirs(self.extracted_tiffs_folder)

    def get_jp2_files(self):
        jp2_files = glob(os.path.join(self.input, '*.jp2'))
        jp2_files.extend(glob(os.path.join(self.input, '*.JP2')))
        jp2_files = sorted(jp2_files)
        if not jp2_files:
            raise FileNotFoundError(f'No JP2 files found in {self.input}')
        return jp2_files

    def run(self):
        print('Running JP2 Reader')
        metadata = self.initialize_info_file()
        resolution_level_folder = os.path.join(
            self.output,
            f'resolution_level_{self.resolution_level}'
        )
        if not os.path.exists(resolution_level_folder):
            os.makedirs(resolution_level_folder)
        with open(os.path.join(resolution_level_folder, settings.INFO_FILE_NAME), 'w') as f:
            f.write(json.dumps(metadata))
        metadata['channel'] = self.channel
        provenance_file_path = os.path.join(self.extracted_tiffs_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, 'w') as f:
            f.write(json.dumps(metadata))
        job_ids = self.extract_tiff_series()
        return provenance_file_path, job_ids

    def inspect_jp2(self):
        inspect_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inspect_jp2.py')
        command = (
            'source /h20/home/lab/miniconda3/bin/activate jp2 && '
            f'python {shlex.quote(inspect_script)} {shlex.quote(self.jp2_files[0])}'
        )
        result = subprocess.run(
            ['bash', '-lc', command],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            error_output = result.stderr.strip() or result.stdout.strip() or 'Unknown JP2 inspection error'
            raise RuntimeError(error_output)
        return json.loads(result.stdout.strip())

    def initialize_info_file(self):
        inspected = self.inspect_jp2()
        self._shape = [len(self.jp2_files), inspected['shape'][0], inspected['shape'][1]]
        options = {
            'out_name': self.output,
            'source': self.input,
            'orientation': 'sal',
            'channels': 1,
            'background_channel': 0,
            'volume_100um_location': '',
            'resolution': [self.resolution_z, self.resolution_y, self.resolution_x],
            'shape': self._shape,
            'resolution_level': self.resolution_level,
            'full_resolution': [self.resolution_z, self.resolution_y, self.resolution_x],
            'full_shape': self._shape,
            'input': {
                'type': 'jp2_series',
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

    def extract_tiff_series(self):
        if self._shape is None:
            self.initialize_info_file()

        expected_tiffs = self._shape[0]
        already_done = glob(os.path.join(self.extracted_tiffs_folder, '*.tif'))
        if len(already_done) == expected_tiffs:
            print('JP2 conversion already completed')
            return []

        path_to_task = os.path.join(self.jobs_folder, 'convert_jp2_to_tiff.sh')
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), 'convert_jp2_to_tiff.py')
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f'#SBATCH -J {self.user}-{self.name}')
            f.write('\n')
            f.write(f'#SBATCH -o {self.jobs_folder}/slurm_%j.out')
            f.write('\n')
            f.write('\n')
            f.write('source /h20/home/lab/miniconda3/bin/activate jp2')
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.extracted_tiffs_folder if ' ' not in self.extracted_tiffs_folder else f'"{self.extracted_tiffs_folder}"')
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        if len(already_done):
            print('Partially processed')
            print('Processed', len(already_done), 'of', expected_tiffs)
            files = os.listdir(self.extracted_tiffs_folder)
            pattern = '_z(\\d+)\\.tif'
            numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
            numbers = set(map(int, numbers))
            job_ids = split_slurm_array(
                path_to_task,
                expected_tiffs,
                numbers,
                partition=settings.SLURM_PARTITION_GPU,
                cores=1,
                memory=256,
                priority=self.priority,
                extra_args=extra_args
            )
        else:
            job_ids = submit_slurm_array(
                path_to_task,
                expected_tiffs,
                partition=settings.SLURM_PARTITION_GPU,
                cores=1,
                memory=256,
                priority=self.priority,
                extra_args=extra_args
            )
        return job_ids
