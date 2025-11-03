import json
import os
import subprocess
import time
from glob import glob

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_job


class dbscan(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "dbscan"
        self.points = kwargs["cell_candidates_path"]
        self.source_provenance_path = os.path.join(os.path.dirname(self.points), f'.{settings.INFO_FILE_NAME}')
        self.source_provenance = json.load(open(self.source_provenance_path, 'r'))
        if not self.input or not self.output:
            self.input = self.source_provenance['base_input_dir']
            self.output = self.source_provenance['base_output_dir']
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.sequence = ",".join([self.source_provenance.get("sequence", ""), self.name])
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']

        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.epsilon = int(kwargs.get("epsilon", 3))
        self.min_samples = int(kwargs.get("min_samples", 2))
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
            f"dbscan_epsilon_{self.epsilon}_minsamples_{self.min_samples}"
        )
        self.out_csv_path = os.path.join(self.save_folder, f"dbscan_{os.path.basename(self.points)}")
        self.prerequisites = kwargs.get('prerequisites', [])
        print("DBSCAN prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running DBSCAN")
        provenance_file_path = self.create_provenance()
        job_ids = []
        if not os.path.exists(self.out_csv_path):
            job_ids = self.run_dbscan()
        else:
            print("Output CSV file already exists")
        return provenance_file_path, job_ids

    def create_provenance(self):
        base_output_dir = self.source_provenance['base_output_dir']
        base_input_dir = self.source_provenance['base_input_dir']
        provenance = {
            "input": {
                "type": "csv",  # input type
                "path": self.points,
            },
            "output": {
                "type": "csv",
                "path": self.out_csv_path,
            },
            "process": {
                "parameters": {
                }
            },
            "source": self.source_provenance_path,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence
        }
        provenance_file_path = os.path.join(os.path.dirname(self.out_csv_path), f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

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

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=12,
            memory=256,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
