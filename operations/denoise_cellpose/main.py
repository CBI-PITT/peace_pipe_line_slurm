import json
import os

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_job
from utils.z_range import normalize_z_range, z_range_provenance, z_range_suffix


class denoise_cellpose(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'denoise_cellpose'
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        if not self.output:
            self.output = self.metadata.get("base_output_dir", self.metadata.get("out_name"))
        self.sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.model = kwargs.get('model', 'denoise_cyto3')
        self.diameter = int(kwargs.get('diameter', 100))
        self.z_selection = normalize_z_range(
            self.metadata,
            kwargs.get('z_start', 0),
            kwargs.get('z_end', -1)
        )
        self.z_start = self.z_selection['start']
        self.z_end = self.z_selection['end']
        self.z_suffix = z_range_suffix(self.z_selection)

        output_operation_folder = os.path.join(self.output, self.name)
        output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, f"slurm_jobs{self.z_suffix}")
        save_folder_name = f"cellpose_model_{self.model}_diameter_{self.diameter}"
        self.save_folder = os.path.join(
            output_folder_sequence,
            f"{save_folder_name}{self.z_suffix}"
        )
        self.save_folder_uint = os.path.join(
            output_folder_sequence,
            f"{save_folder_name}_uint{self.z_suffix}"
        )
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Denoise prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if not os.path.exists(self.save_folder_uint):
            os.makedirs(self.save_folder_uint)

    def run(self):
        print("Running cellpose in chunks")
        provenance_file_path = self.create_provenance()
        job_ids = self.run_denoising()
        return provenance_file_path, job_ids

    def create_provenance(self):
        source = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        base_output_dir = self.metadata['base_output_dir']
        base_input_dir = self.metadata['base_input_dir']
        provenance = {
            "input": {
                "type": "tiff_series",
                "path": self.input,
            },
            "output": {
                "type": "tiff_series",
                "path": self.save_folder_uint,
            },
            "process": {
                "parameters": {
                    "model": self.model,
                    "diameter": self.diameter,
                    "z_start": self.z_selection['requested_start'],
                    "z_end": self.z_selection['requested_end'],
                    "raw_normalization_scope": "all_available_input_z",
                    "denoised_normalization_scope": "selected_output_z"
                }
            },
            "source": source,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "resolution": self.metadata['resolution'],
            "shape": self.metadata['shape'],
            "orientation": self.metadata['orientation'],
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence,
            "processed_z_range": z_range_provenance(self.z_selection)
        }
        provenance_file_path = os.path.join(self.save_folder_uint, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def run_denoising(self):
        path_to_task = os.path.join(self.jobs_folder, f"run_all_steps.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "run_all_steps.py")
        print("self output", self.output)
        path_parts = self.output.split('/')
        if self.user in path_parts:
            user_pos = path_parts.index(self.user)
            experiment = "_".join(path_parts[user_pos + 1:])
        else:
            experiment = os.path.basename(self.output.rstrip('/'))
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-denoise-cellpose-main")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script} ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.save_folder if ' ' not in self.save_folder else f'"{self.save_folder}"')
            f.write(' ')
            f.write(self.save_folder_uint if ' ' not in self.save_folder_uint else f'"{self.save_folder_uint}"')
            f.write(' ')
            f.write(
                f'{self.resolution_level} {self.channel} {self.user} {self.priority} {self.model} {str(self.diameter)}'
            )
            f.write(' ')
            f.write(f'{experiment}')
            f.write(' ')
            f.write(f'{self.z_start} {self.z_end}')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=64,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
