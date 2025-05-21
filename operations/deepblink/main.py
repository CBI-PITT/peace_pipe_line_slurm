import json
import os
import subprocess

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_job


class deepblink(ImageOperation):
    """
    Deep learning based spot detection.

    Works on a folder with tif/tiff files that represent a z-stack.
    If this folder was obtained from Imaris, it extracts chunks directly from .ims file

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage/resolution_level_0/channel_1",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "deepblink",
    }
    """
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        if not self.output:
            self.output = self.metadata.get("base_output_dir", self.metadata.get("out_name"))
        self.signal_channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.with_dbscan = kwargs.get('with_dbscan', False)
        self.output_operation_folder = os.path.join(self.output, 'deepblink')
        self.chunks_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "deepblink_chunks")
        self.jobs_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "slurm_jobs")
        self.detection_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "detection")
        self.napari_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "detection_napari")
        self.out_csv_name = "merged_dbscan_df.csv" if self.with_dbscan else "merged_df.csv"
        self.out_csv_path = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", self.out_csv_name)
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Deepblink prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.chunks_folder):
            os.makedirs(self.chunks_folder)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)
        if not os.path.exists(self.napari_folder):
            os.makedirs(self.napari_folder)
        self.dbscan_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "dbscan")
        if self.with_dbscan and not os.path.exists(self.dbscan_folder):
            os.makedirs(self.dbscan_folder)

    def run(self):
        print("Running deepblink in chunks")
        provenance_file_path = self.create_provenance()
        job_ids = []
        if not os.path.exists(self.out_csv_path):
            job_ids = self.run_all()
        else:
            print("Output CSV file already exists")
        return provenance_file_path, job_ids

    def run_all(self):
        path_to_task = os.path.join(self.jobs_folder, f"run_all_steps.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "run_all_steps.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-deepblink-main")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script} ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.output if ' ' not in self.output else f'"{self.output}"')
            f.write(' ')
            f.write(f'{self.resolution_level} {self.signal_channel} {self.user} {int(self.with_dbscan)} {self.priority}')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=1,
            memory=128,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids

    def create_provenance(self):
        sequence = ",".join([self.metadata.get("sequence", ""), "deepblink"])
        provenance = {
            "input": {
                "type": "tiff_series",  # input type
                "path": self.input,
            },
            "output": {
                "type": "csv",
                "path": self.out_csv_path,
            },
            "process": {
                "parameters": {
                    "with_dbscan": self.with_dbscan,
                }
            },
            "source": os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'),  # input provenance file
            "channel": self.signal_channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": self.metadata.get('base_output_dir', self.metadata['out_name']),
            "base_input_dir": self.input,
            "sequence": sequence
        }
        provenance_file_path = os.path.join(
            self.output_operation_folder,
            f'resolution_level_{self.resolution_level}',
            f"channel_{self.signal_channel}",
            f'.{settings.INFO_FILE_NAME}'
        )
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))

        return provenance_file_path
