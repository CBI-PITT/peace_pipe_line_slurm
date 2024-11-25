import os
import subprocess

from imaris_ims_file_reader import ims
import numpy as np
import dask
import dask.array as da


CHUNK_SIZE = (40, 1700, 3500)


class deepblink:
    """
    Deep learning based spot detection.

    PEACE JSON example: (name should start with 'SLURM_settings_'
    {
        "input": "/h20/Public/cakir-i/4CL16/chow1_mag8x_montage.ims",
        "output": "/h20/Public/cakir-i/4CL16/analysis/chow1_mag8x_montage",
        "operation": "deepblink",
        "extras": {
            "signal_channel": 0,
            "resolution_level": 0
        }
    }
    defaults:
    signal_channel = 0
    resolution_level = 0
    """
    def __init__(self, input, output, **kwargs):
        self.input = input
        self.output = output
        self.signal_channel = int(kwargs.get('signal_channel', 0))
        self.resolution_level = int(kwargs.get('resolution_level', 0))
        self.chunks_folder = os.path.join(self.output, f"resolution_level_{self.resolution_level}", "deepblink_chunks")
        self.jobs_folder = os.path.join(self.output, f"resolution_level_{self.resolution_level}", "slurm_jobs")
        self.detection_folder = os.path.join(self.output, f"resolution_level_{self.resolution_level}", "detection")
        self.napari_folder = os.path.join(self.output, f"resolution_level_{self.resolution_level}", "detection_napari")
        if not os.path.exists(self.chunks_folder):
            os.makedirs(self.chunks_folder)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.detection_folder):
            os.makedirs(self.detection_folder)
        if not os.path.exists(self.napari_folder):
            os.makedirs(self.napari_folder)

    def run(self):
        print("Running deepblink in chunks")
        print("Input", self.input)
        print("Output", self.output)
        print("Channel", self.signal_channel)
        number_of_chunks = self.get_chunking()
        self.submit_detection_cpu_slurm_array(number_of_chunks)

    def get_chunking(self):
        ims_file = ims(self.input)
        tiff_stack_shape = list(ims_file.metaData[self.resolution_level, 0, self.signal_channel, 'shape'][-3:])
        resolution = ims_file.metaData[self.resolution_level, 0, self.signal_channel, 'resolution']
        print("Shape", tiff_stack_shape)
        print("Resolution", resolution)
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
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script} {self.input} {self.output} {self.resolution_level} {self.signal_channel} $SLURM_ARRAY_TASK_ID')
            f.write('\n')

        # run it on compute (cpu) partition
        # command = ['sbatch', f'--array=11-20', '-p', 'compute', '--mem=32Gb', '-n12', '-o', '/h20/CBI/Iana/json/slurm_out', path_to_task]
        command = ['sbatch', f'--array=0-{number_of_chunks}', '-p', 'compute', '--mem=32Gb', '-n12', path_to_task]
        print("command", command)
        subprocess.run(command)



