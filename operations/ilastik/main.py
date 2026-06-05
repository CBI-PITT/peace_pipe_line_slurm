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
from utils.slurm import submit_slurm_array, split_slurm_array


class ilastik(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "ilastik"
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        if not self.output:
            self.output = self.metadata.get("base_output_dir", self.metadata.get("out_name"))
        self.sequence = ",".join([self.metadata.get('sequence', ''), self.name])

        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.model = kwargs['model_path']  # TODO no default model
        self.binarize_threshold = kwargs.get('binarize_threshold', 0.5)

        output_operation_folder = os.path.join(self.output, self.name)
        self.output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(self.output_folder_sequence, "slurm_jobs")
        self.save_folder = os.path.join(
            self.output_folder_sequence,
            f"ilastik_model_{os.path.basename(self.model).replace('.ilp', '')}"
        )
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Ilastik prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        binary_save_folder = os.path.join(
            self.output_folder_sequence,
            f"ilastik_model_{os.path.basename(self.model).replace('.ilp', '')}_threshold_{self.binarize_threshold}"
        )
        if not os.path.exists(binary_save_folder):
            os.makedirs(binary_save_folder)

    def run(self):
        print("Running ilastik")
        provenance_file_path = self.create_provenance()
        job_ids = self.do_segmentation()
        return provenance_file_path, job_ids

    def create_provenance(self):
        base_output_dir = self.metadata['base_output_dir'] if 'base_output_dir' in self.metadata else self.metadata['out_name']
        base_input_dir = self.metadata['base_input_dir'] if 'base_input_dir' in self.metadata else self.input
        provenance = {
            "input": {
                "type": "tiff_series",  # input type
                "path": self.input,
            },
            "output": {
                "type": "tiff_series",
                "path": self.save_folder,
            },
            "process": {
                "parameters": {
                    "model_path": self.model,
                }
            },
            "source": os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'),  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def do_segmentation(self):
        z_layers = self.metadata['shape'][-3]
        path_to_task = os.path.join(self.jobs_folder, f"ilastik_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_ilastik.py")
        from utils.containers import build_container_exec_prefix

        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-ilastik")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_ilastik_%A_%a.out")
            f.write('\n')
            f.write('\n')
            f.write(build_container_exec_prefix('ilastik', os.path.dirname(main_script)))
            f.write(f' python {slurm_script}')
            f.write(' ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.output_folder_sequence if ' ' not in self.output_folder_sequence else f'"{self.output_folder_sequence}"')
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write(str(self.model))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write(' ')
            f.write(str(self.binarize_threshold))
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        already_done = glob(os.path.join(self.save_folder, "*.tif"))
        if len(already_done):
            print("Partially processed")
            print("Processed", len(already_done), "of", z_layers)
            files = os.listdir(self.save_folder)
            pattern = "_z(\d+)\.tif"
            numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
            numbers = set(map(int, numbers))
            job_ids = split_slurm_array(
                path_to_task,
                z_layers,
                numbers,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=12,
                memory=128,
                priority=self.priority,
                extra_args=extra_args
            )
        else:
            job_ids = submit_slurm_array(
                path_to_task,
                z_layers,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=12,
                memory=128,
                priority=self.priority,
                extra_args=extra_args
            )
        return job_ids
