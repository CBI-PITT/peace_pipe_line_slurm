import json
import os
import subprocess
import time
from glob import glob

from ..base import ImageOperation
from analysis import settings


class resnet_classification(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.cells = kwargs['cell_candidates_path']
        self.model_path = kwargs['model_path']
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']

        self.output_operation_folder = os.path.join(self.output, 'resnet_classification')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.extracted_tiffs_folder = os.path.join(self.output, f'resolution_level_{self.resolution_level}', f'channel_{self.channel}')
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}',
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running ResNet classification")
        print("Input", self.input)
        print("Output", self.output)
        print("Channel", self.channel)
        print("Resolution level", self.resolution_level)
        self.run_classification()

    def run_classification(self):
        path_to_task = os.path.join(self.jobs_folder, f"resnet_main_cpu_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_classification.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.save_folder if ' ' not in self.save_folder else f'"{self.save_folder}"')
            f.write(' ')
            f.write(self.cells if ' ' not in self.cells else f'"{self.cells}"')
            f.write(' ')
            f.write(self.model_path if ' ' not in self.model_path else f'"{self.model_path}"')
            f.write('\n')

        #### run it on compute (cpu) partition
        command = ['sbatch', '-p', 'compute', '--mem=64Gb', '-n24', path_to_task]
        subprocess.run(command)
