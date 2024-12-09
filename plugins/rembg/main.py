import os
import subprocess
from glob import glob

from imaris_ims_file_reader import ims

from operations.base import ImageOperation


class rembg(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.channel = int(kwargs.get('channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))

        self.output_operation_folder = os.path.join(self.output, 'rembg')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.extracted_tiffs_folder = os.path.join(
            self.output,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}'
        )
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f"r{self.resolution_level}_c{self.channel}_removed_background"
        )
        self.ims_file = ims(self.input)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running rmbg")
        self.extract_tiff_series()  # extract tiff series in SLURM
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
        while len(extracted_files) < z_layers:
            extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
            print(f"Extracted files: {len(extracted_files)} of {z_layers}")
            time.sleep(10)
        self.run_rembg()

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

    def run_rembg(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"rembg_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_rembg.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate rembg")
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

