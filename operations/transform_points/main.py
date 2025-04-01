import os
import subprocess
import json

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class transform_points(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata_path = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        self.metadata = json.load(open(self.metadata_path, 'r'))
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.registration_path = kwargs.get('registration_path')  # path to folder
        self.cells_path = kwargs.get('cells_path')  # CSV
        self.output_operation_folder = os.path.join(self.output, 'transform_points')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.results_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def run(self):
        self.get_df()

    def get_df(self):
        path_to_task = os.path.join(self.jobs_folder, f"transform_points.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "get_structures_df.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-transform-points")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate cellfinder")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.cells_path if ' ' not in self.cells_path else f'"{self.cells_path}"')
            f.write(' ')
            f.write(self.registration_path if ' ' not in self.registration_path else f'"{self.registration_path}"')
            f.write(' ')
            f.write(self.results_folder if ' ' not in self.results_folder else f'"{self.results_folder}"')
            f.write(' ')
            f.write(self.metadata_path if ' ' not in self.metadata_path else f'"{self.metadata_path}"')
            f.write('\n')

        submit_slurm_job(
            path_to_task,
            partition=settings.SLURM_PARTITION_HIGH_RAM,
            cores=8,
            memory=128,
            priority=self.priority
        )
        # command = [
        #     'sbatch',
        #     '-p', settings.SLURM_PARTITION_HIGH_RAM,
        #     '--mem=128Gb',
        #     '-n8',
        #     f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
        #     path_to_task
        # ]
        # subprocess.run(command)
