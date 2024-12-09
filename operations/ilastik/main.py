import os
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from ..base import ImageOperation


class ilastik(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.channel = int(kwargs.get('channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))
        self.model = kwargs['model_path']  # TODO no default model

        self.output_operation_folder = os.path.join(self.output, 'ilastik')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.extracted_tiffs_folder = os.path.join(self.output, f'resolution_level_{self.resolution_level}', f'channel_{self.channel}')
        self.save_folder = os.path.join(self.output_operation_folder, f"r{self.resolution_level}_c{self.channel}_ilastik_model_{os.path.basename(self.model)}")
        self.ims_file = ims(self.input)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running ilastik")
        print("Input", self.input)
        print("Output", self.output)
        print("Channel", self.channel)
        print("Resolution level", self.resolution_level)
        self.extract_tiff_series()  # extract tiff series in SLURM
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
        while len(extracted_files) < z_layers:
            extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
            print(f"Extracted files: {len(extracted_files)} of {z_layers}")
            time.sleep(10)
        self.do_segmentation()

    def extract_tiff_series(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"extract_imaris_z_layers_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "extract_imaris_z_layer.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(str(self.input))
            f.write(' ')
            f.write(str(self.output))
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

    def do_segmentation(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"ilastik_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_ilastik.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate ilastik")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(str(self.input))
            f.write(' ')
            f.write(str(self.output))
            f.write(' ')
            f.write(str(self.resolution_level))
            f.write(' ')
            f.write(str(self.channel))
            f.write(' ')
            f.write(str(self.model))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        #### run it on compute (cpu) partition
        command = ['sbatch', f'--array=0-{z_layers-1}', '-p', 'compute', '--mem=32Gb', '-n12', path_to_task]
        subprocess.run(command)
