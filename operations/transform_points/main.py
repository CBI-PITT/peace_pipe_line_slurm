import os
import subprocess
import json

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_job


class transform_points(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "transform_points"
        self.registration_path = kwargs.get('registration_path')  # path to folder
        self.cells_path = kwargs.get('cells_path')  # CSV
        self.source_provenance_path = os.path.join(os.path.dirname(self.cells_path), f'.{settings.INFO_FILE_NAME}')
        self.source_provenance = json.load(open(self.source_provenance_path, 'r'))
        if not self.input or not self.output:
            self.input = self.source_provenance['base_input_dir']
            self.output = self.source_provenance['base_output_dir']
        self.metadata_path = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        self.metadata = json.load(open(self.metadata_path, 'r'))
        self.sequence = ",".join([self.metadata.get('sequence', ''), self.name])
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        output_operation_folder = os.path.join(self.output, self.name)
        output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
        self.results_folder = output_folder_sequence
        self.out_csv_path = os.path.join(self.results_folder, f"{os.path.basename(self.cells_path.replace('.csv', ''))}_for_dashboard.csv")
        self.out_binary_tiff_path = os.path.join(self.results_folder, f"{os.path.basename(self.cells_path.replace('.csv', ''))}_for_dashboard_binary.tiff")
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
        if not os.path.exists(self.out_csv_path) or not os.path.exists(self.out_binary_tiff_path):
            job_ids = self.get_df()
        else:
            print("Output CSV file and binary TIFF already exist")
        return provenance_file_path, job_ids

    def create_provenance(self):
        base_output_dir = self.source_provenance['base_output_dir']
        base_input_dir = self.source_provenance['base_input_dir']
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
