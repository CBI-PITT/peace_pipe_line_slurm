import os
import subprocess
import json

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class transform_points(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.registration_path = kwargs.get('registration_path')  # path to folder
        self.cells_path = kwargs.get('cells_path')  # CSV
        if not self.input or not self.output:
            provenance_path = os.path.join(os.path.dirname(self.cells_path), f'.{settings.INFO_FILE_NAME}')
            provenance = json.load(open(provenance_path, 'r'))
            self.input = provenance['base_input_dir']
            self.output = provenance['base_output_dir']
        self.metadata_path = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        self.metadata = json.load(open(self.metadata_path, 'r'))
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.output_operation_folder = os.path.join(self.output, 'transform_points')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.results_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
        )
        self.out_csv_path = os.path.join(self.results_folder, f"{os.path.basename(self.cells_path.replace('.csv', ''))}_for_dashboard.csv")
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def run(self):
        provenance_file_path = self.create_provenance()
        job_ids = []
        if not os.path.exists(self.out_csv_path):
            job_ids = self.get_df()
        else:
            print("Output CSV file already exists")
        return provenance_file_path, job_ids

    def create_provenance(self):
        source = os.path.join(os.path.dirname(self.cells_path), f'.{settings.INFO_FILE_NAME}')
        source_provenance = json.load(open(source, 'r'))
        base_output_dir = source_provenance['base_output_dir']
        base_input_dir = source_provenance['base_input_dir']
        sequence = ",".join([self.metadata.get('sequence', ''), 'transform_points'])
        provenance = {
            "input": [
                {
                    "type": "csv",
                    "path": self.cells_path,
                },
                {
                    "type": "folder",
                    "path": self.registration_path,
                }
            ],
            "output": {
                "type": "csv",
                "path": self.out_csv_path,
            },
            "process": {
                "parameters": {
                }
            },
            "source": source,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": sequence
        }
        provenance_file_path = os.path.join(os.path.dirname(self.out_csv_path), f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

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

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=settings.SLURM_PARTITION_HIGH_RAM,
            cores=8,
            memory=128,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
