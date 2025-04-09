import json
import os
import subprocess

from ..base import ImageOperation
from analysis import settings
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
        self.user = kwargs.get('user', 'lab')
        self.priority = kwargs.get('priority', '2')
        self.with_dbscan = kwargs.get('with_dbscan', False)
        self.output_operation_folder = os.path.join(self.output, 'deepblink')
        self.chunks_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "deepblink_chunks")
        self.jobs_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "slurm_jobs")
        self.detection_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "detection")
        self.napari_folder = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", "detection_napari")
        self.out_csv_name = "merged_dbscan_df.csv" if self.with_dbscan else "merged_df.csv"
        self.out_csv_path = os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", self.out_csv_name)
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
        self.create_provenance()
        if not os.path.exists(self.out_csv_path):
            self.run_all()
        else:
            print("Output CSV file already exists")

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

        submit_slurm_job(
            path_to_task,
            partition=f'{settings.SLURM_PARTITION_HIGH_RAM}',
            cores=8,
            memory=128,
            priority=self.priority
        )

    def create_provenance(self):
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
            "base_output_dir": self.metadata['out_name'],
            "base_input_dir": self.input
        }
        with open(
                os.path.join(
                    self.output,
                    f'resolution_level_{self.resolution_level}',
                    f"channel_{self.signal_channel}",
                    f'.{settings.INFO_FILE_NAME}'
                ),
                "w") as f:
            f.write(json.dumps(provenance))


    # def get_chunking(self):
    #     tiff_stack_shape = self.metadata['shape']
    #     ratios = (np.array(tiff_stack_shape) / np.array(CHUNK_SIZE)).astype('int') + 1
    #     patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
    #     print("patchify_chunks_shape", patchify_chunks_shape)
    #     origin_coords = self.get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
    #     chunk_indices = self.get_chunk_indices(origin_coords, CHUNK_SIZE)
    #     print("Total chunks", len(chunk_indices))
    #     np.save(os.path.join(self.chunks_folder, 'origin_coords.npy'), origin_coords)
    #     np.save(os.path.join(self.chunks_folder, 'chunk_indices.npy'), chunk_indices)
    #     return len(chunk_indices)

    # @staticmethod
    # def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    #     """
    #     Get coordinates of each chunk origin.
    #     """
    #     coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
    #     coords = np.empty(coords_shape, dtype=np.uint16)
    #     print(" coords shape", coords.shape)
    #     for z in range(coords.shape[0]):
    #         for y in range(coords.shape[1]):
    #             for x in range(coords.shape[2]):
    #                 coords[z, y, x, :] = np.array((
    #                     z * chunk_size[0],
    #                     y * chunk_size[1],
    #                     x * chunk_size[2]
    #                 ))
    #     coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
    #     print("final coords shape", coords.shape)
    #     return coords
    #
    # @staticmethod
    # def get_chunk_indices(origin_coords, chunk_size):
    #     indices = []
    #     for origin in list(origin_coords):
    #         indices.append([
    #             slice(origin[0], origin[0] + chunk_size[0], 1),
    #             slice(origin[1], origin[1] + chunk_size[1], 1),
    #             slice(origin[2], origin[2] + chunk_size[2], 1)
    #         ])
    #     return indices

    # def submit_detection_cpu_slurm_array(self, number_of_chunks):
    #     # write slurm job
    #     path_to_task = os.path.join(self.jobs_folder, f"cpu_array_all_chunks.sh")
    #     main_script = os.path.abspath(__file__)
    #     slurm_script = os.path.join(os.path.dirname(main_script), "process_one_chunk.py")
    #     with open(path_to_task, 'w') as f:
    #         f.write('#!/bin/bash\n')
    #         f.write('\n')
    #         f.write(f"#SBATCH -J {self.user}-deepblink-cpu")
    #         f.write('\n')
    #         f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
    #         f.write('\n')
    #         f.write('\n')
    #         f.write("source /h20/home/lab/miniconda3/bin/activate peace")
    #         f.write('\n')
    #         f.write(f'python {slurm_script} ')
    #         f.write(self.input if ' ' not in self.input else f'"{self.input}"')
    #         f.write(' ')
    #         f.write(self.output if ' ' not in self.output else f'"{self.output}"')
    #         f.write(' ')
    #         f.write(f'{self.resolution_level} {self.signal_channel} {self.user} $SLURM_ARRAY_TASK_ID')
    #         f.write('\n')
    #
    #     # run it on compute (cpu) partition
    #     command = [
    #         'sbatch',
    #         f'--array=0-{number_of_chunks}',
    #         '-p', settings.SLURM_PARTITION_CPU,
    #         '--mem=32Gb',
    #         '-n12',
    #         f'--nice={settings.SLURM_JOBS_NICE_LEVEL}',
    #         path_to_task
    #     ]
    #     # print("command", command)
    #     subprocess.run(command)

    # def merge_df(self):  # TODO lunch as a separate process on a GPU node
    #     df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']
    #     df = pd.DataFrame(columns=df_column_names)
    #     csv_files = sorted(glob(os.path.join(self.napari_folder, 'napari*.csv')))
    #     print("CSV files", len(csv_files), csv_files[:3])
    #     origin_coords = np.load(os.path.join(self.chunks_folder, 'origin_coords.npy'), allow_pickle=True)
    #     for chunk_file in csv_files:
    #         current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
    #         print("Processing", current_chunk)
    #         chunk_df = pd.read_csv(chunk_file)
    #         if chunk_df.empty:
    #             continue
    #         chunk_df_corrected = pd.DataFrame()
    #         z_values = chunk_df[['axis-0']].to_numpy()
    #         y_values = chunk_df[['axis-1']].to_numpy()
    #         x_values = chunk_df[['axis-2']].to_numpy()
    #         z_values += origin_coords[current_chunk, 0]
    #         y_values += origin_coords[current_chunk, 1]
    #         x_values += origin_coords[current_chunk, 2]
    #         chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
    #         chunk_df_corrected['axis-0'] = z_values
    #         chunk_df_corrected['axis-1'] = y_values
    #         chunk_df_corrected['axis-2'] = x_values
    #         df = pd.concat([df, chunk_df_corrected])
    #
    #     print("Saving df")
    #     df.to_csv(os.path.join(self.output_operation_folder, f"resolution_level_{self.resolution_level}", f"channel_{self.signal_channel}", 'merged_df.csv'))
