import json
import os

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import shell_arg, submit_slurm_job


ATLAS_COLUMNS = ["atlas_structure_name", "atlas_structure_acronym", "atlas_structure_number"]


def csv_header_has_columns(csv_path, columns):
    try:
        with open(csv_path, 'r') as f:
            header = f.readline()
    except OSError:
        return False
    header_columns = [x.strip() for x in header.strip().split(',')]
    return all(column in header_columns for column in columns)


class nearest_neighbor(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "nearest_neighbor"
        self.cells_path = kwargs.get('cells_path')  # CSV
        self.source_provenance_path = os.path.join(os.path.dirname(self.cells_path), f'.{settings.INFO_FILE_NAME}')
        self.source_provenance = json.load(open(self.source_provenance_path, 'r'))
        if not self.input or not self.output:
            self.input = self.source_provenance['base_input_dir']
            self.output = self.source_provenance['base_output_dir']
        self.sequence = ",".join([self.source_provenance.get('sequence', ''), self.name])
        self.channel = int(self.source_provenance.get('channel', 0))
        self.resolution_level = int(self.source_provenance.get('resolution_level', 0))
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
        self.out_nn_csv_path = os.path.join(self.results_folder, f"{os.path.basename(self.cells_path.replace('.csv', ''))}_nn_distances.csv")
        # atlas-region columns decide whether a per-region summary CSV is produced
        self.has_atlas_columns = csv_header_has_columns(self.cells_path, ATLAS_COLUMNS)
        self.out_region_csv_path = None
        if self.has_atlas_columns:
            self.out_region_csv_path = os.path.join(self.results_folder, f"{os.path.basename(self.cells_path.replace('.csv', ''))}_nn_region_stats.csv")
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Nearest neighbor prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def run(self):
        provenance_file_path = self.create_provenance()
        job_ids = []
        expected_outputs = [self.out_nn_csv_path]
        if self.out_region_csv_path:
            expected_outputs.append(self.out_region_csv_path)
        if not all(os.path.exists(path) for path in expected_outputs):
            job_ids = self.get_nn_distances()
        else:
            print("Nearest-neighbor output CSV files already exist")
        return provenance_file_path, job_ids

    def create_provenance(self):
        base_output_dir = self.source_provenance['base_output_dir']
        base_input_dir = self.source_provenance['base_input_dir']
        provenance = {
            "input": {
                "type": "csv",
                "path": self.cells_path,
            },
            "output": {
                "type": "csv",
                "path": self.out_nn_csv_path,
            },
            "process": {
                "parameters": {
                    "with_region_stats": self.has_atlas_columns
                }
            },
            "source": self.source_provenance_path,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence
        }
        provenance_file_path = os.path.join(os.path.dirname(self.out_nn_csv_path), f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def get_nn_distances(self):
        path_to_task = os.path.join(self.jobs_folder, "nearest_neighbor.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_nn.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {shell_arg(self.user + '-nearest-neighbor')}")
            f.write('\n')
            f.write(f"#SBATCH -o {shell_arg(self.jobs_folder + '/slurm_%j.out')}")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate dbscan")
            f.write('\n')
            f.write(f'python {shell_arg(slurm_script)}')
            f.write(' ')
            f.write(shell_arg(self.cells_path))
            f.write(' ')
            f.write(shell_arg(self.results_folder))
            f.write(' ')
            f.write(shell_arg(self.source_provenance_path))
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=settings.SLURM_PARTITION_HIGH_RAM,
            cores=8,
            memory=64,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
