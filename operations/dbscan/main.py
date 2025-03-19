import json
import os
import subprocess
import time
from glob import glob

from ..base import ImageOperation
from analysis import settings


class dbscan(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']

        self.user = kwargs.get('user', 'lab')
        self.points = kwargs["cell_candidates_path"]
        self.epsilon = int(kwargs.get("epsilon", 3))
        self.min_samples = int(kwargs.get("min_samples", 2))
        self.output_operation_folder = os.path.join(self.output, 'dbscan')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}',
            f"dbscan_epsilon_{self.epsilon}_minsamples_{self.min_samples}"
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running DBSCAN")
        # print("Input", self.input)
        # print("Output", self.output)
        # print("Points", self.points)
        self.run_dbscan()

    def run_dbscan(self):
        path_to_task = os.path.join(self.jobs_folder, f"dbscan_{self.epsilon}_{self.min_samples}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_dbscan.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-dbscan")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate dbscan")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.points if ' ' not in self.points else f'"{self.points}"')
            f.write(' ')
            f.write(self.save_folder if ' ' not in self.save_folder else f'"{self.save_folder}"')
            f.write(' ')
            f.write(str(self.epsilon))
            f.write(' ')
            f.write(str(self.min_samples))
            f.write('\n')

        #### run it on compute (cpu) partition
        command = [
            'sbatch',
            '-p', settings.SLURM_PARTITION_HIGH_RAM,
            '--mem=256Gb',
            '-n24',
            f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
            path_to_task
        ]
        subprocess.run(command)
