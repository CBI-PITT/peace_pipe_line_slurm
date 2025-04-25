import json
import os
import subprocess
import time
from glob import glob

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class resnet_classification(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "resnet_classification"
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.cells = kwargs['cell_candidates_path']
        self.model_path = kwargs['model_path']
        self.metadata = json.load(open(os.path.join(os.path.dirname(self.cells), f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.input = self.metadata['base_input_dir']
        self.output = self.metadata['base_output_dir']
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']
        self.output_operation_folder = os.path.join(self.output, self.name)
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.extracted_tiffs_folder = os.path.join(self.output, f'resolution_level_{self.resolution_level}', f'channel_{self.channel}')
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}',
        )
        self.out_csv_path = os.path.join(self.save_folder, f'predicted_cells_{os.path.basename(self.model_path)}.csv')
        self.prerequisites = kwargs.get('prerequisites', [])
        print("ResNet prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running ResNet classification")
        provenance_file_path = self.create_provenance()
        job_ids = []
        if not os.path.exists(self.out_csv_path):
            job_ids = self.run_classification()
        else:
            print("Output CSV file already exists")
        return provenance_file_path, job_ids

    def create_provenance(self):
        source = os.path.join(os.path.dirname(self.cells), f'.{settings.INFO_FILE_NAME}')
        source_provenance = json.load(open(source, 'r'))
        base_output_dir = source_provenance['base_output_dir']
        base_input_dir = source_provenance['base_input_dir']
        sequence = ",".join([source_provenance.get("sequence", ""), self.name])
        provenance = {
            "input": {
                "type": "csv",  # input type
                "path": self.cells,
            },
            "output": {
                "type": "csv",
                "path": self.out_csv_path,
            },
            "process": {
                "parameters": {
                    "model": self.model_path
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

    def run_classification(self):
        path_to_task = os.path.join(self.jobs_folder, f"resnet_main_cpu_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_classification.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-resnet-cpu")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
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
            f.write(self.cells if ' ' not in self.cells else f'"{self.cells}"')
            f.write(' ')
            f.write(self.model_path if ' ' not in self.model_path else f'"{self.model_path}"')
            f.write(' ')
            f.write(self.user)
            f.write(' ')
            f.write(self.priority)
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_CPU}',
            cores=24,
            memory=64,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
