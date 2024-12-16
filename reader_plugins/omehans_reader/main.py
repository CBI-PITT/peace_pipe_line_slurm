import json
import os
import subprocess

from operations.base import ImageReader


class omehans_reader(ImageReader):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.channel = int(kwargs.get('channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))

        self.jobs_folder = os.path.join(self.output, "slurm_jobs")
        self.extracted_tiffs_folder = os.path.join(
            self.output,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}'
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.extracted_tiffs_folder):
            os.makedirs(self.extracted_tiffs_folder)

    def run(self):
        print("Running OMEhans Reader")
        self.extract_tiff_series()

    def extract_tiff_series(self):
        # metadata = json.load(open(os.path.join(self.input, '.zattrs'), 'r'))
        # resolution_levels = len(metadata['multiscales'][0]['datasets'])
        array_metadata = json.load(open(os.path.join(self.input, f"scale{self.resolution_level}", '.zarray'), 'r'))
        z_layers = array_metadata['shape'][-3]
        path_to_task = os.path.join(self.jobs_folder, f"extract_omehans_z_layers_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "extract_z_layer.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate omehans-reader")  # TODO create a separate env?
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(str(self.input if ' ' not in self.input else f'"{self.input}"'))
            f.write(' ')
            f.write(str(self.output if ' ' not in self.output else f'"{self.output}"'))
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        #### run it on compute (cpu) partition
        command = ['sbatch', f'--array=0-{z_layers-1}', '-p', 'compute', '--mem=32Gb', '-n12', path_to_task]
        subprocess.run(command)