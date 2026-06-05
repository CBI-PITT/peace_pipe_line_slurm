import json
import os
import re
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from analysis import settings
from operations.base import ImageReader
from utils import get_user
from utils.slurm import submit_partial_slurm_array


class imaris_reader_crop(ImageReader):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "imaris_reader_crop"
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.channel = int(kwargs.get('channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))
        self.ims_file = ims(self.input)
        self.channels = self.ims_file.Channels
        self.output_operation_folder = os.path.join(self.output, self.name)
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        shape = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3:]
        self.z_start = kwargs.get('z_start', 0)
        self.z_end = kwargs.get('z_end', shape[0])
        if self.z_end == -1:
            self.z_end = shape[0]
        self.y_start = kwargs.get('y_start', 0)
        self.y_end = kwargs.get('y_end', shape[1])
        if self.y_end == -1:
            self.y_end = shape[1]
        self.x_start = kwargs.get('x_start', 0)
        self.x_end = kwargs.get('x_end', shape[2])
        if self.x_end == -1:
            self.x_end = shape[2]

        self.extracted_tiffs_folders = [
            os.path.join(
                self.output_operation_folder,
                f'resolution_level_{self.resolution_level}',
                f'channel_{self.channel}',
                f"z_{self.z_start}-{self.z_end}_y_{self.y_start}-{self.y_end}_x_{self.x_start}-{self.x_end}"
            )
        ]
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        for extracted_tiffs_folder in self.extracted_tiffs_folders:
            if not os.path.exists(extracted_tiffs_folder):
                os.makedirs(extracted_tiffs_folder)

    def run(self):
        print("Running Imaris Reader")
        # create info json files with metadata
        metadata = self.initialize_info_file()
        with open(os.path.join(self.output_operation_folder, f'resolution_level_{self.resolution_level}', settings.INFO_FILE_NAME), "w") as f:
            f.write(json.dumps(metadata))

        metadata['channel'] = self.channel
        with open(os.path.join(self.extracted_tiffs_folders[0], f".{settings.INFO_FILE_NAME}"), "w") as f:
            f.write(json.dumps(metadata))
            f.close()
        job_ids = self.extract_tiff_series()
        with open(os.path.join(self.extracted_tiffs_folders[0], f".{settings.JOBS_FILE_NAME}"), "w") as f:
            f.write(json.dumps(job_ids))

        return os.path.join(self.extracted_tiffs_folders[0], f".{settings.INFO_FILE_NAME}"), job_ids  # return path to provenance and job ids

    def extract_tiff_series(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"extract_imaris_z_layers_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "extract_imaris_z_layer.py")
        from utils.containers import build_container_exec_prefix

        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-extract-tiffs")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write(build_container_exec_prefix('peace', os.path.dirname(main_script)))
            f.write(f' python {slurm_script}')
            f.write(' ')
            f.write(str(self.input if ' ' not in self.input else f'"{self.input}"'))
            f.write(' ')
            f.write(str(self.extracted_tiffs_folders[0] if ' ' not in self.extracted_tiffs_folders[0] else f'"{self.extracted_tiffs_folders[0]}"'))
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write(' ')
            f.write(str(self.y_start))
            f.write(' ')
            f.write(str(self.y_end))
            f.write(' ')
            f.write(str(self.x_start))
            f.write(' ')
            f.write(str(self.x_end))
            f.write('\n')

        job_ids = submit_partial_slurm_array(
            path_to_task,
            self.z_start,
            self.z_end,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=8,
            priority=self.priority
        )
        return job_ids

    def initialize_info_file(self):
        options = {
            "out_name": self.output,
            "source": self.ims_file.filePathComplete,
            "channels": self.ims_file.Channels,
            "volume_100um_location": os.path.join(
                self.output,
                self.ims_file.fileName + settings.SUFFIX_100UM_VOLUME
            ),
            "resolution": self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'resolution')][-3:],
            "shape": self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3:],
            "resolution_level": self.resolution_level,
            "full_resolution": self.ims_file.resolution,
            "full_shape": self.ims_file.shape,
            "input": {
                "type": "ims",
                "path": self.ims_file.filePathComplete
            },
            "output": {
                "type": "tiff_series",
                "path": self.extracted_tiffs_folders[0]
            },
            "process": {},
            "base_output_dir": self.output,
            "base_input_dir": "",
            "sequence": self.name
        }
        return options
