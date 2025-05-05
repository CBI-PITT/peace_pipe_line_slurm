import json
import os
import subprocess
import time
from glob import glob

import numpy as np

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_array


CHUNK_SIZE = settings.DEEPBLINK_CHUNK_SIZE


class unet_3d(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "unet_3d"
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.channel = int(self.metadata['channel'])
        self.resolution_level = int(self.metadata['resolution_level'])
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.model = kwargs.get('model')  # TODO: add default model

        self.output_operation_folder = os.path.join(self.output, self.name)
        self.chunks_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.channel}", "chunks")
        self.jobs_folder = os.path.join(self.output_operation_folder, "slurm_jobs")
        self.save_folder = os.path.join(
            self.output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"model_{os.path.basename(self.model)}"
        )
        self.prerequisites = kwargs.get('prerequisites', [])
        print("3D U-Net prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.chunks_folder):
            os.makedirs(self.chunks_folder)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def run(self):
        print("Running 3D UNet")
        provenance_file_path = self.create_provenance()
        number_of_chunks = self.get_chunking()
        print("number of chunks", number_of_chunks)
        job_ids = self.submit_detection_cpu_slurm_array(number_of_chunks)
        return provenance_file_path, job_ids

    def create_provenance(self):
        sequence = ",".join([self.metadata.get('sequence', ''), self.name])
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
                    "model_path": self.model,
                }
            },
            "source": os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'),  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": self.metadata['out_name'],
            "base_input_dir": self.input,
            "sequence": sequence
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def get_chunking(self):
        tiff_stack_shape = self.metadata['shape']
        ratios = (np.array(tiff_stack_shape) / np.array(CHUNK_SIZE)).astype('int') + 1
        patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
        print("patchify_chunks_shape", patchify_chunks_shape)
        origin_coords = self.get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
        chunk_indices = self.get_chunk_indices(origin_coords, CHUNK_SIZE)
        print("Total chunks", len(chunk_indices))
        np.save(os.path.join(self.chunks_folder, 'origin_coords.npy'), origin_coords)
        np.save(os.path.join(self.chunks_folder, 'chunk_indices.npy'), chunk_indices)
        return len(chunk_indices)

    @staticmethod
    def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
        """
        Get coordinates of each chunk origin.
        """
        coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
        coords = np.empty(coords_shape, dtype=np.uint16)
        print(" coords shape", coords.shape)
        for z in range(coords.shape[0]):
            for y in range(coords.shape[1]):
                for x in range(coords.shape[2]):
                    coords[z, y, x, :] = np.array((
                        z * chunk_size[0],
                        y * chunk_size[1],
                        x * chunk_size[2]
                    ))
        coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
        print("final coords shape", coords.shape)
        return coords

    @staticmethod
    def get_chunk_indices(origin_coords, chunk_size):
        indices = []
        for origin in list(origin_coords):
            indices.append([
                slice(origin[0], origin[0] + chunk_size[0], 1),
                slice(origin[1], origin[1] + chunk_size[1], 1),
                slice(origin[2], origin[2] + chunk_size[2], 1)
            ])
        return indices

    def submit_detection_cpu_slurm_array(self, number_of_chunks):
        # write slurm job
        path_to_task = os.path.join(self.jobs_folder, f"cpu_array_all_chunks.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "process_one_chunk.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-3d-unet-cpu")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write(f"source {str(settings.HOME)}/miniconda3/bin/activate peace")  # TODO more general path
            f.write('\n')
            f.write(f'python {slurm_script} ')
            f.write(self.input if ' ' not in self.input else f'"{self.input}"')
            f.write(' ')
            f.write(self.output if ' ' not in self.output else f'"{self.output}"')
            f.write(' ')
            f.write(f'{self.resolution_level} {self.channel} $SLURM_ARRAY_TASK_ID {self.model} {self.user} {self.priority}')
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_array(
            path_to_task,
            number_of_chunks,
            partition=f'{settings.SLURM_PARTITION_CPU}',
            cores=12,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
