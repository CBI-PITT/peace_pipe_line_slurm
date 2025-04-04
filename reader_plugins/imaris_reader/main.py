import json
import os
import subprocess
import time
from glob import glob

from imaris_ims_file_reader import ims

from analysis import settings
from analysis.guess_background_channel import guess_background
from analysis.guess_brain_orientation import guess_orientation
from operations.base import ImageReader
from utils.slurm import submit_slurm_array


class imaris_reader(ImageReader):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.channel = int(kwargs.get('channel', 0))
        self.all_channels = int(kwargs.get('all_channels', False))
        self.resolution_level = int(kwargs.get('resolution_level', 0))
        self.ims_file = ims(self.input)
        self.channels = self.ims_file.Channels
        self.jobs_folder = os.path.join(self.output, "slurm_jobs")
        if self.all_channels:
            self.extracted_tiffs_folders = [
                os.path.join(self.output, f'resolution_level_{self.resolution_level}', f'channel_{x}')
                for x in range(self.channels)
            ]
        else:
            self.extracted_tiffs_folders = [
                os.path.join(self.output, f'resolution_level_{self.resolution_level}', f'channel_{self.channel}')
            ]
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        for extracted_tiffs_folder in self.extracted_tiffs_folders:
            if not os.path.exists(extracted_tiffs_folder):
                os.makedirs(extracted_tiffs_folder)

    def run(self):
        print("Running Imaris Reader")
        # create info json files with metadata
        metadata = self.initialize_info_file()
        with open(os.path.join(self.output, f'resolution_level_{self.resolution_level}', settings.INFO_FILE_NAME), "w") as f:
            f.write(json.dumps(metadata))
        if self.all_channels:
            for channel in range(self.channels):
                metadata['channel'] = channel
                with open(os.path.join(self.extracted_tiffs_folders[channel], f".{settings.INFO_FILE_NAME}"), "w") as f:
                    f.write(json.dumps(metadata))
            self.extract_tiff_series_all_channels()
        else:
            metadata['channel'] = self.channel
            with open(os.path.join(self.extracted_tiffs_folders[0], f".{settings.INFO_FILE_NAME}"), "w") as f:
                f.write(json.dumps(metadata))
            # extract tiff series in SLURM
            self.extract_tiff_series()

        # # check extraction progress
        # z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        # extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
        # while len(extracted_files) < z_layers:
        #     extracted_files = glob(os.path.join(self.extracted_tiffs_folder, "*.tif"))
        #     print(f"Extracted files: {len(extracted_files)} of {z_layers}")
        #     time.sleep(10)

    def extract_tiff_series(self):
        z_layers = self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3]
        path_to_task = os.path.join(self.jobs_folder, f"extract_imaris_z_layers_rl{self.resolution_level}_c{self.channel}.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "extract_imaris_z_layer.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-extract-tiffs")
            f.write('\n')
            f.write(f"#SBATCH -o {self.output}/slurm_jobs/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")  # TODO create a separate env?
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

        submit_slurm_array(
            path_to_task,
            z_layers,
            partition=f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=12,
            memory=32,
            priority=self.priority
        )
        # nice_value = settings.PRIORITY_TO_NICE_MAP_COMPUTE[self.priority]
        # command = [
        #     'sbatch',
        #     f'--array=0-{z_layers-1}',
        #     '-p', f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
        #     '--mem=32Gb',
        #     '-n12',
        #     f'--nice={nice_value}',
        #     path_to_task
        # ]
        # subprocess.run(command)

    def extract_tiff_series_all_channels(self):
        for channel in range(self.channels):
            z_layers = self.ims_file.metaData[(self.resolution_level, 0, channel, 'shape')][-3]
            path_to_task = os.path.join(self.jobs_folder, f"extract_imaris_z_layers_rl{self.resolution_level}_c{channel}.sh")
            main_script = os.path.abspath(__file__)
            slurm_script = os.path.join(os.path.dirname(main_script), "extract_imaris_z_layer.py")
            with open(path_to_task, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('\n')
                f.write(f"#SBATCH -J {self.user}-extract-tiffs")
                f.write('\n')
                f.write(f"#SBATCH -o {self.output}/slurm_jobs/slurm_extract_tiffs_%j.out")
                f.write('\n')
                f.write('\n')
                f.write("source /h20/home/lab/miniconda3/bin/activate peace")  # TODO create a separate env?
                f.write('\n')
                f.write(f'python {slurm_script}')
                f.write(' ')
                f.write(str(self.input if ' ' not in self.input else f'"{self.input}"'))
                f.write(' ')
                f.write(str(self.output if ' ' not in self.output else f'"{self.output}"'))
                f.write(' ')
                f.write(str(self.resolution_level))
                f.write(' ')
                f.write(str(channel))
                f.write(' ')
                f.write('$SLURM_ARRAY_TASK_ID')
                f.write('\n')

            submit_slurm_array(
                path_to_task,
                z_layers,
                partition=f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
                cores=12,
                memory=32,
                priority=self.priority
            )
            # command = [
            #     'sbatch',
            #     f'--array=0-{z_layers-1}',
            #     '-p', f'{settings.SLURM_PARTITION_CPU},{settings.SLURM_PARTITION_HIGH_RAM}',
            #     '--mem=32Gb',
            #     '-n12',
            #     f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
            #     path_to_task
            # ]
            # subprocess.run(command)

    def initialize_info_file(self):
        orientation = guess_orientation(self.ims_file, save_100um_volume=True, out_dir=self.output)
        background_channel, background_channel_nmi = guess_background(
            self.ims_file,
            save_100um_volume=True,
            out_dir=self.output
        )
        options = {
            "out_name": self.output,
            "source": self.ims_file.filePathComplete,
            "orientation": orientation,
            "channels": self.ims_file.Channels,
            "background_channel": background_channel,
            "volume_100um_location": os.path.join(
                self.output,
                self.ims_file.fileName + settings.SUFFIX_100UM_VOLUME
            ),
            "resolution": self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'resolution')][-3:],
            "shape": self.ims_file.metaData[(self.resolution_level, 0, self.channel, 'shape')][-3:],
            "resolution_level": self.resolution_level,
            "full_resolution": self.ims_file.resolution,
            "full_shape": self.ims_file.shape
        }
        return options

