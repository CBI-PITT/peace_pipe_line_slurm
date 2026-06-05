import json
import os
import re
import subprocess
from glob import glob

from imaris_ims_file_reader import ims

from operations.base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import split_slurm_array, submit_slurm_array


class rembg(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')

        self.output_operation_folder = os.path.join(self.output, 'rembg')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        if self.metadata.get('sequence'):
            previous_operations = self.metadata['sequence'].split(',')
            previous_operation = f"_{previous_operations[-1]}" if len(previous_operations) > 1 else ""
        else:
            previous_operation = ""
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}',
            f"removed_background{previous_operation}"
        )
        self.prerequisites = kwargs.get('prerequisites', [])
        print("rembg prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running rembg")
        provenance_file_path = self.create_provenance()
        job_ids = self.run_rembg()
        return provenance_file_path, job_ids

    def create_provenance(self):
        source = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        source_provenance = json.load(open(source, 'r'))
        base_output_dir = source_provenance['base_output_dir']
        base_input_dir = source_provenance['base_input_dir']
        sequence = ",".join([source_provenance.get("sequence", ""), "rembg"])
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
                }
            },
            "source": source,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "resolution": source_provenance['resolution'],
            "shape": source_provenance['shape'],
            "orientation": source_provenance['orientation'],
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": sequence
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def run_rembg(self):
        z_layers = self.metadata['shape'][-3]
        path_to_task = os.path.join(self.jobs_folder, f"rembg_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_rembg.py")
        from utils.containers import build_container_exec_prefix

        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-rembg")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write(build_container_exec_prefix('rembg', os.path.dirname(main_script)))
            f.write(f' python {slurm_script}')
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
            f.write('\n')

        extra_args = {'--export': 'OMP_NUM_THREADS=12'}
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
                partition=settings.SLURM_PARTITION_CPU,
                cores=12,
                memory=32,
                priority=self.priority,
                extra_args=extra_args
            )
        else:
            job_ids = submit_slurm_array(
                path_to_task,
                z_layers,
                partition=settings.SLURM_PARTITION_CPU,
                cores=12,
                memory=32,
                priority=self.priority,
                extra_args=extra_args
            )
        return job_ids
