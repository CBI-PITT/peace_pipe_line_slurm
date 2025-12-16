"""
This operation wants
- input (folder with per-slice tiffs)
- output (generic folder with all analysis for the current 3D image)

It creates 'attenuation_correction' subfolder within the output folder - all results go there.

The correction is done using following algorithm:
- estimate background in each slice via grey-level morphological opening,
- compute its mean and std per slice,
- apply a linear transform to each slice so that the background stats match a chosen reference slice.
"""

import json
import os
import re
import subprocess
import time
from glob import glob

import numpy as np

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_job


class attenuation_correction(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'attenuation_correction'
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.reference_z = kwargs.get('reference_z', 0)
        self.opening_radius = int(kwargs.get('opening_radius', 5))

        output_operation_folder = os.path.join(self.output, self.name)
        output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
        self.save_folder = os.path.join(
            output_folder_sequence,
            f"attenuation_correction_ref_{self.reference_z}_radius_{self.opening_radius}"
        )
        # self.save_folder_uint = self.save_folder + "_uint"
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Attenuation correction prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        # if not os.path.exists(self.save_folder_uint):
            # os.makedirs(self.save_folder_uint)

    def run(self):
        print("Running attenuation correction")
        provenance_file_path = self.create_provenance()
        job_ids = self.run_correction()
        return provenance_file_path, job_ids

    def create_provenance(self):
        source = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        base_output_dir = self.metadata['base_output_dir']
        base_input_dir = self.metadata['base_input_dir']
        provenance = {
            "input": {
                "type": "tiff_series",
                "path": self.input,
            },
            "output": {
                "type": "tiff_series",
                "path": self.save_folder,
            },
            "process": {
                "parameters": {
                    "reference_z": self.reference_z,
                    "opening_radius": self.opening_radius
                }
            },
            "source": source,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "resolution": self.metadata['resolution'],
            "shape": self.metadata['shape'],
            "orientation": self.metadata['orientation'],
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def run_correction(self):
        path_to_task = os.path.join(self.jobs_folder, f"run_all_steps.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "run_all_steps.py")
        path_parts = self.output.split('/')
        user_pos = path_parts.index(self.user)
        experiment = "_".join(path_parts[user_pos + 1:])
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-attenuation-correction-main")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/attenuation_correction_main_slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script} ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.save_folder if ' ' not in self.save_folder else f'"{self.save_folder}"')
            f.write(' ')
            f.write(
                f'{self.resolution_level} {self.channel} {self.user} {self.priority} {self.reference_z} {str(self.opening_radius)}'
            )
            f.write(' ')
            f.write(f'{experiment}')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=16,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids