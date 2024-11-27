import os
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims


class cellfinder:
    """
    Thresholding-based cell detection.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "cellfinder",
        "extras": {
            "signal_channel": 0,
            "resolution_level": 0
        }
    }
    """
    def __init__(self, input, output, **kwargs):
        print("kwargs", kwargs)
        self.input = input
        self.output = output
        self.signal_channel = int(kwargs.get('signal_channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))

        self.output_operation_folder = os.path.join(self.output, 'cellfinder')
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.detection_folder = os.path.join(self.output_operation_folder, f"cellfinder_channel_{self.signal_channel}")
        self.ims_file = ims(self.input)
        self.resolution = self.ims_file.metaData[(self.resolution_level, 0, self.signal_channel, 'resolution')][-3:]
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)

    def run(self):
        print("Running brainreg")
        print("Input", self.input)
        print("Output", self.output)
        print("Channel", self.signal_channel)
        self.stack_to_detect = os.path.join(self.output_operation_folder, f'resolution_level_{self.resolution_level}', f'channel_{self.signal_channel}')
        self.extract_tiff_series()  # extract tiff series in SLURM
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.signal_channel, 'shape')][-3]
        extracted_files = glob(os.path.join(self.output_operation_folder, f'resolution_level_{self.resolution_level}', f'channel_{self.signal_channel}', "*.tif"))
        while len(extracted_files) < z_layers:
            extracted_files = glob(os.path.join(self.output_operation_folder, f'resolution_level_{self.resolution_level}', f'channel_{self.signal_channel}', "*.tif"))
            print(f"Extracted files: {len(extracted_files)} of {z_layers}")
            time.sleep(10)
        self.run_detection()  # run brainreg in SLURM
        # delete tiff series  # TODO

    def extract_tiff_series(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.signal_channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"extract_imaris_z_layers_rl{self.resolution_level}_c{self.signal_channel}.sh")
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
            f.write(str(self.signal_channel))
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write('\n')

        #### run it on compute (cpu) partition
        command = ['sbatch', f'--array=0-{z_layers-1}', '-p', 'compute', '--mem=32Gb', '-n12', path_to_task]
        subprocess.run(command)

    def run_detection(self):
        path_to_task = os.path.join(self.jobs_folder, f"cellfinder_rl{self.resolution_level}_c{self.signal_channel}.sh")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            # f.write('ulimit -n 600000')
            # f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate cellfinder")
            f.write('\n')
            f.write(f'cellfinder -s {self.stack_to_detect} -b {self.stack_to_detect} -o {self.detection_folder}')
            f.write(f' -v {str(self.resolution[0])} {str(self.resolution[1])} {str(self.resolution[2])}')
            f.write(f' --orientation sal --atlas allen_mouse_25um')
            f.write(f' --no-analyse --no-figures --no-register --no-classification --ball-z-size {str(int(self.resolution[0])+1)}')
            f.write('\n')

        print("Starting cellfinder detection...")
        #### run on gpu partition
        command = ['sbatch', '-p', 'gpu', '--gres=gpu:1', '--mem=64Gb', '-n8', path_to_task]
        # command = ['sbatch', '-p', 'compute', '--mem=64Gb', '-n24', path_to_task]  # for CPU partition - does not work
        subprocess.run(command)

