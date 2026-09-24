import json
import os

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import shell_arg, submit_slurm_job


def shell_path(path):
    return shell_arg(path)


def resolutions_close(a, b, rtol=0.001):
    try:
        if a is None or b is None or len(a) != len(b):
            return False
        return all(
            abs(float(x) - float(y)) <= rtol * max(abs(float(x)), abs(float(y)))
            for x, y in zip(a, b)
        )
    except (TypeError, ValueError):
        return False


def shapes_close(a, b):
    try:
        if a is None or b is None or len(a) < 3 or len(b) < 3:
            return False
        return [int(a[-2]), int(a[-1])] == [int(b[-2]), int(b[-1])]
    except (TypeError, ValueError):
        return False


def is_provenance_file(path):
    return (
        isinstance(path, str)
        and path != ""
        and os.path.basename(path) == f'.{settings.INFO_FILE_NAME}'
        and os.path.isfile(path)
    )


def resolve_chain_value(source_provenance, key):
    # value from the provenance itself or the nearest ancestor that has it
    value = source_provenance.get(key)
    source_path = source_provenance.get('source')
    seen = set()
    while value is None and is_provenance_file(source_path) and source_path not in seen:
        seen.add(source_path)
        with open(source_path, 'r') as f:
            ancestor = json.load(f)
        value = ancestor.get(key)
        source_path = ancestor.get('source')
    return value


def resolve_raw_provenance(source_provenance_path):
    # follow provenance source links; reader nodes point source at non-JSON
    # paths (.ims file, .zarr folder, plain folder), so the walk ends at the
    # raw reader provenance
    last_path = source_provenance_path
    seen = {last_path}
    with open(last_path, 'r') as f:
        last = json.load(f)
    source_path = last.get('source')
    while is_provenance_file(source_path) and source_path not in seen:
        seen.add(source_path)
        last_path = source_path
        with open(source_path, 'r') as f:
            last = json.load(f)
        source_path = last.get('source')
    return last_path, last


def folder_has_tiffs(folder):
    if not os.path.isdir(folder):
        return False
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.tif', '.tiff')):
            return True
    return False


def resolve_raw_tiff_folder(raw_provenance, raw_provenance_path, channel):
    candidates = []
    output = raw_provenance.get('output')
    if isinstance(output, dict) and output.get('path'):
        candidates.append(output['path'])
    candidates.append(os.path.dirname(raw_provenance_path))
    base_output_dir = raw_provenance.get('base_output_dir')
    if base_output_dir:
        candidates.append(os.path.join(
            base_output_dir,
            f"resolution_level_{raw_provenance.get('resolution_level', 0)}",
            f"channel_{channel}"
        ))
    for candidate in candidates:
        if folder_has_tiffs(candidate):
            return candidate
    raise FileNotFoundError(
        f'Could not locate the raw TIFF series folder from the cells CSV provenance '
        f'chain; tried {candidates}'
    )


class mean_intensity(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = "mean_intensity"
        self.cells_path = kwargs.get('cells_path')  # CSV
        self.radius = float(kwargs.get('radius', 5))  # um
        self.source_provenance_path = os.path.join(os.path.dirname(self.cells_path), f'.{settings.INFO_FILE_NAME}')
        self.source_provenance = json.load(open(self.source_provenance_path, 'r'))
        if not self.input or not self.output:
            self.input = self.source_provenance['base_input_dir']
            self.output = self.source_provenance['base_output_dir']
        self.sequence = ",".join([self.source_provenance.get('sequence', ''), self.name])
        self.channel = int(self.source_provenance.get('channel', 0))
        self.resolution_level = int(self.source_provenance.get('resolution_level', 0))
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        # resolve the original raw (reader-produced) TIFF series from the
        # provenance chain, skipping any pre-processed TIFF series
        self.raw_provenance_path, self.raw_provenance = resolve_raw_provenance(self.source_provenance_path)
        self.chain_resolution = resolve_chain_value(self.source_provenance, 'resolution')
        self.chain_shape = resolve_chain_value(self.source_provenance, 'shape')
        self.raw_tiff_folder = resolve_raw_tiff_folder(self.raw_provenance, self.raw_provenance_path, self.channel)
        self.validate_sources()
        self.radius_str = f"{self.radius:g}"
        output_folder_sequence = os.path.join(
            self.output,
            self.name,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence},mean_intensity_r{self.radius_str}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
        self.results_folder = output_folder_sequence
        self.partials_folder = os.path.join(output_folder_sequence, "partial_dfs")
        csv_basename = os.path.basename(self.cells_path.replace('.csv', ''))
        self.out_csv_path = os.path.join(self.results_folder, f"{csv_basename}_mean_intensity_r{self.radius_str}.csv")
        self.out_column = f"mean_intensity_r{self.radius_str}"
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Mean intensity prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.partials_folder):
            os.makedirs(self.partials_folder)

    def validate_sources(self):
        raw_level = self.raw_provenance.get('resolution_level')
        if raw_level is None or int(raw_level) != self.resolution_level:
            raise ValueError(
                f"Raw image provenance resolution level ({raw_level}) does not match "
                f"the detection resolution level ({self.resolution_level})"
            )
        raw_resolution = self.raw_provenance.get('resolution')
        if not resolutions_close(raw_resolution, self.chain_resolution):
            raise ValueError(
                f"Raw image resolution {raw_resolution} does not match the detection "
                f"chain resolution {self.chain_resolution}; coordinates may not be comparable"
            )
        raw_shape = self.raw_provenance.get('shape')
        if not shapes_close(raw_shape, self.chain_shape):
            raise ValueError(
                f"Raw image shape {raw_shape} does not match the detection chain shape "
                f"{self.chain_shape} in x/y; coordinates may not be comparable"
            )

    def run(self):
        provenance_file_path = self.create_provenance()
        job_ids = []
        if not os.path.exists(self.out_csv_path):
            job_ids = self.submit_orchestrator()
        else:
            print("Output CSV file already exists")
        return provenance_file_path, job_ids

    def create_provenance(self):
        base_output_dir = self.source_provenance['base_output_dir']
        base_input_dir = self.source_provenance['base_input_dir']
        provenance = {
            "input": {
                "type": "csv",
                "path": self.cells_path,
            },
            "output": {
                "type": "csv",
                "path": self.out_csv_path,
            },
            "process": {
                "parameters": {
                    "radius_um": self.radius,
                    "column": self.out_column,
                    "raw_source": self.raw_tiff_folder,
                }
            },
            "source": self.source_provenance_path,  # input provenance file
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence
        }
        if self.chain_resolution is not None:
            provenance["resolution"] = self.chain_resolution
        if self.chain_shape is not None:
            provenance["shape"] = self.chain_shape
        provenance_file_path = os.path.join(self.results_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def submit_orchestrator(self):
        path_to_task = os.path.join(self.jobs_folder, "mean_intensity_orchestrator.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_mean_intensity.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {shell_arg(self.user + '-mean-intensity')}")
            f.write('\n')
            f.write(f"#SBATCH -o {shell_arg(self.jobs_folder + '/slurm_%j.out')}")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {shell_arg(slurm_script)}')
            f.write(' ')
            f.write(shell_path(self.cells_path))
            f.write(' ')
            f.write(shell_path(self.raw_tiff_folder))
            f.write(' ')
            f.write(shell_path(self.raw_provenance_path))
            f.write(' ')
            f.write(shell_path(self.results_folder))
            f.write(' ')
            f.write(str(self.radius))
            f.write(' ')
            f.write(self.user)
            f.write(' ')
            f.write(self.priority)
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        job_ids = submit_slurm_job(
            path_to_task,
            partition=settings.SLURM_PARTITION_CPU,
            cores=8,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )
        return job_ids
