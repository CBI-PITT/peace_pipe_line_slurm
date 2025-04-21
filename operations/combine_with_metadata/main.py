import os
import subprocess
import json

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_job


class combine_with_metadata(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
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
        self.metadata_fields = kwargs.get('metadata', [])
        self.output_operation_folder = os.path.join(self.output, 'combine_with_metadata')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.results_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
        )
        self.out_csv_path = os.path.join(self.results_folder, f"{os.path.basename(self.cells_path).replace('.csv', '_with_metadata.csv')}")
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def run(self):
        self.create_provenance()
        self.update_df()

    def create_provenance(self):
        source = os.path.join(os.path.dirname(self.cells_path), f'.{settings.INFO_FILE_NAME}')
        source_provenance = json.load(open(source, 'r'))
        base_output_dir = source_provenance['base_output_dir']
        base_input_dir = source_provenance['base_input_dir']
        sequence = ",".join([source_provenance.get("sequence", ""), "combine_with_metadata"])
        provenance = {
            "input": {
                "type": "csv",
                "path": self.cells_path,
            },
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
        with open(
                os.path.join(os.path.dirname(self.out_csv_path), f'.{settings.INFO_FILE_NAME}'),
                "w") as f:
            f.write(json.dumps(provenance))

    def update_df(self):
        path_to_task = os.path.join(self.jobs_folder, f"combine_with_metadata.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "combine.py")
        print("metadata fields", self.metadata_fields)
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-combine-with-metadata")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_combine_w_metadata_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.cells_path if ' ' not in self.cells_path else f'"{self.cells_path}"')
            f.write(' ')
            f.write(self.results_folder if ' ' not in self.results_folder else f'"{self.results_folder}"')
            f.write(' ')
            f.write(self.metadata_path if ' ' not in self.metadata_path else f'"{self.metadata_path}"')
            f.write(' ')
            for metadata_field in self.metadata_fields:
                f.write(f"{metadata_field['key'].replace(' ', '')}={metadata_field['value'].replace(' ', '')}")
                f.write(' ')
            f.write('\n')

        submit_slurm_job(
            path_to_task,
            partition=settings.SLURM_PARTITION_HIGH_RAM,
            cores=8,
            memory=128,
            priority=self.priority
        )
