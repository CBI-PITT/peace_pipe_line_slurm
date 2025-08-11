import json
import os
import re
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_job


class spotiflow(ImageOperation):
    """
    input: tiff series
    output: csv file
    """
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'spotiflow'
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.model = kwargs.get('model', 'general')
        self.with_dbscan = kwargs.get('with_dbscan', False)

        output_operation_folder = os.path.join(self.output, self.name)
        output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
        self.save_folder = os.path.join(output_folder_sequence, f'{self.model}_model')
        self.out_csv_name = f"{self.model}_merged_dbscan_df.csv" if self.with_dbscan else f"{self.model}_merged_df.csv"
        self.out_csv_path = os.path.join(output_folder_sequence, self.out_csv_name)
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Spotiflow prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running spotiflow")
        provenance_file_path = self.create_provenance()
        job_ids = self.do_operation()
        return provenance_file_path, job_ids

    def create_provenance(self):
        base_output_dir = self.metadata['base_output_dir']
        base_input_dir = self.metadata['base_input_dir']
        provenance = {
            "input": {
                "type": "tiff_series",
                "path": self.input
            },
            "output": {
                "type": "csv",
                "path": self.out_csv_path
            },
            "process": {
                "parameters": {
                    "with_dbscan": self.with_dbscan,
                    "model": self.model
                }
            },
            "source": os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'),  # input provenance file
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

    def do_operation(self):
        path_to_task = os.path.join(self.jobs_folder, f"run_all_steps.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "run_all_steps.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-spotiflow-main")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/spotiflow_main_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script} ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.save_folder if ' ' not in self.save_folder else f'"{self.save_folder}"')
            f.write(' ')
            f.write(f'{self.user} {self.priority} {self.model} {int(self.with_dbscan)}')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=1,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
