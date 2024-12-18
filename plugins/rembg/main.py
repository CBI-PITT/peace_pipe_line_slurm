import json
import os
import subprocess
from glob import glob

from imaris_ims_file_reader import ims

from operations.base import ImageOperation
from analysis import settings


class rembg(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']

        self.output_operation_folder = os.path.join(self.output, 'rembg')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f'channel_{self.channel}',
            f"removed_background"
        )
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running rembg")
        # self.extract_tiff_series()  # extract tiff series in SLURM
        # z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        # extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
        # while len(extracted_files) < z_layers:
        #     extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
        #     print(f"Extracted files: {len(extracted_files)} of {z_layers}")
        #     time.sleep(10)
        self.run_rembg()

    def run_rembg(self):
        z_layers = self.metadata['shape'][-3]
        path_to_task = os.path.join(self.jobs_folder, f"rembg_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_rembg.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate rembg")
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
            f.write('\n')

        #### run it on compute (cpu) partition
        command = ['sbatch', f'--array=0-{z_layers-1}', '--export=OMP_NUM_THREADS=12', '-p', 'compute', '--mem=32Gb', '-n12', path_to_task]
        subprocess.run(command)

