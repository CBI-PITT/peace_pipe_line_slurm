import json
import os
import re
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from ..base import ImageOperation
from analysis import settings
from utils.slurm import submit_slurm_array, split_slurm_array


class ilastik(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        if not self.output:
            self.output = self.metadata.get("base_output_dir", self.metadata.get("out_name"))
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.model = kwargs['model_path']  # TODO no default model

        self.output_operation_folder = os.path.join(self.output, 'ilastik')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        # self.extracted_tiffs_folder = os.path.join(self.output, f'resolution_level_{self.resolution_level}', f'channel_{self.channel}')
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"ilastik_model_{os.path.basename(self.model).replace('.ilp', '')}"
        )
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running ilastik")
        self.create_provenance()
        self.do_segmentation()

    def create_provenance(self):
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
            "base_output_dir": self.metadata['out_name'],
            "base_input_dir": self.input
        }
        with open(
                os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}'),
                "w") as f:
            f.write(json.dumps(provenance))

    def do_segmentation(self):
        z_layers = self.metadata['shape'][-3]
        path_to_task = os.path.join(self.jobs_folder, f"ilastik_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_ilastik.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-ilastik")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_ilastik_%A_%a.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate ilastik")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.output if ' ' not in self.output else f'"{self.output}"')
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write(str(self.model))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        already_done = glob(os.path.join(self.save_folder, "*.tif"))
        if len(already_done):
            print("Partially processed")
            print("Processed", len(already_done), "of", z_layers)
            files = os.listdir(self.save_folder)
            pattern = "_z(\d+)\.tif"
            numbers = [re.findall(pattern, x)[0] for x in files]
            numbers = set(map(int, numbers))
            split_slurm_array(
                path_to_task,
                z_layers,
                numbers,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=12,
                memory=32,
                priority=self.priority
            )
        else:
            submit_slurm_array(
                path_to_task,
                z_layers,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=12,
                memory=32,
                priority=self.priority
            )
