import json
import os

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_indices
from utils.z_range import existing_z_indices, normalize_z_range, z_range_provenance, z_range_suffix


class gaussian_blur(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'gaussian_blur'
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.sigma = kwargs.get('sigma', 1.0)
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
        self.save_folder = os.path.join(
            output_folder_sequence,
            f"gaussian_blurred_sigma_{self.sigma}{self.z_suffix}"
        )
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Gaussian blur prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running gaussian blur")
        provenance_file_path = self.create_provenance()
        job_ids = self.do_gaussian_blur()
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
                "path": self.save_folder,
            },
            "process": {
                "parameters": {
                    "sigma": self.sigma,
                    "z_start": self.z_selection['requested_start'],
                    "z_end": self.z_selection['requested_end']
                }
            },
            "source": source,
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
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def do_gaussian_blur(self):
        path_to_task = os.path.join(self.jobs_folder, f"gaussian_blur_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_gaussian_blur.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-gaussian-blur")
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
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write(' ')
            f.write(str(self.sigma))
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        return submit_slurm_indices(
            path_to_task,
            range(self.z_start, self.z_end),
            existing_outputs=existing_z_indices(self.save_folder),
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )
