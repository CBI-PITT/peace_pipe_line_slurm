import json
import os
import subprocess
import time
from glob import glob

from ..base import ImageOperation
from analysis import settings


class delete_background_detections(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']

        self.user = kwargs.get('user', 'lab')
        self.points = kwargs["cell_candidates_path"]
        self.masks = kwargs["fg_mask_path"]

        self.output_operation_folder = os.path.join(self.output, 'delete_background_detections')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}',
            f"cleaned_bg_{os.path.basename(self.points).replace('.csv', '')}"
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running delete_background_detections")
        # print("Input", self.input)
        # print("Output", self.output)
        # print("Points", self.points)
        self.run_all_z_layers()

    def run_all_z_layers(self):
        path_to_task = os.path.join(self.jobs_folder, f"delete_background_detections_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "process_all_z_layers.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-delete-bg-detections-main")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/main_delete_bg_detections_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.save_folder if ' ' not in self.save_folder else f'"{self.save_folder}"')
            f.write(' ')
            f.write(self.points if ' ' not in self.points else f'"{self.points}"')
            f.write(' ')
            f.write(self.masks if ' ' not in self.masks else f'"{self.masks}"')
            f.write(' ')
            f.write(self.output if ' ' not in self.output else f'"{self.output}"')
            f.write(' ')
            f.write(self.user)
            f.write('\n')

        command = [
            'sbatch',
            '-p', f'{settings.SLURM_PARTITION_HIGH_RAM}',
            '--mem=32Gb',
            '-n12',
            f'--nice=0',
            # f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
            path_to_task
        ]
        subprocess.run(command)
