import json
import os
import re
from glob import glob

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import split_slurm_array, submit_slurm_array


class resize_image(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'resize_image'
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.scale_factor = float(kwargs.get('scale_factor', 1.0))

        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be greater than 0")

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
            f"resized_{self.scale_factor}"
        )
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Resize image prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running image resize")
        provenance_file_path = self.create_provenance()
        job_ids = self.do_resize_image()
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
                    "scale_factor": self.scale_factor,
                    "anti_aliasing_gray": True,
                    "anti_aliasing_binary": False
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
            "sequence": self.sequence
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def do_resize_image(self):
        z_layers = self.metadata['shape'][-3]
        path_to_task = os.path.join(self.jobs_folder, f"resize_image_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_resize_image.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-resize-image")
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
            f.write(str(self.scale_factor))
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
            pattern = "_z(\\d+)\\.tif"
            numbers = [re.findall(pattern, x)[0] for x in files if x.endswith('.tif')]
            numbers = set(map(int, numbers))
            job_ids = split_slurm_array(
                path_to_task,
                z_layers,
                numbers,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=1,
                memory=32,
                priority=self.priority,
                extra_args=extra_args
            )
        else:
            job_ids = submit_slurm_array(
                path_to_task,
                z_layers,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=1,
                memory=32,
                priority=self.priority,
                extra_args=extra_args
            )
        return job_ids
