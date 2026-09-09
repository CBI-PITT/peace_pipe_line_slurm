import json
import os
from glob import glob

import tifffile

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.slurm import submit_slurm_indices
from utils.z_range import (
    Z_FILENAME_PATTERN,
    get_available_z_range,
    normalize_z_range,
    z_range_provenance,
    z_range_suffix,
)


def get_tiff_files(folder):
    files = glob(os.path.join(folder, "*.tif"))
    files.extend(glob(os.path.join(folder, "*.tiff")))
    return sorted(files)


def parse_scalar_operand(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def canonical_tiff_path(folder, metadata, z):
    resolution_level = int(metadata['resolution_level'])
    channel = int(metadata['channel'])
    filename = f"r{resolution_level:02d}_t00_c{channel:02d}_z{z:04d}.tif"
    return os.path.join(folder, filename)


def index_tiff_files(folder, metadata, allow_expected=False):
    files = get_tiff_files(folder)
    if files:
        parsed = []
        for path in files:
            match = Z_FILENAME_PATTERN.search(os.path.basename(path))
            parsed.append((path, int(match.group(1)) if match else None))

        parsed_count = sum(z is not None for _, z in parsed)
        if parsed_count == len(parsed):
            files_by_z = {}
            for path, z in parsed:
                if z in files_by_z:
                    raise ValueError(f"Multiple TIFF files represent z index {z} in {folder}")
                files_by_z[z] = path
            if allow_expected:
                available_start, available_end = get_available_z_range(metadata)
                for z in range(available_start, available_end):
                    files_by_z.setdefault(z, canonical_tiff_path(folder, metadata, z))
            return files_by_z
        if parsed_count:
            raise ValueError(f"Cannot mix indexed and unindexed TIFF filenames in {folder}")

        available_start, available_end = get_available_z_range(metadata)
        if len(files) != available_end - available_start:
            raise ValueError(
                f"Cannot assign absolute z indices to {len(files)} TIFF files in {folder}"
            )
        return dict(zip(range(available_start, available_end), files))

    if allow_expected:
        available_start, available_end = get_available_z_range(metadata)
        return {
            z: canonical_tiff_path(folder, metadata, z)
            for z in range(available_start, available_end)
        }
    return {}


class image_calculator(ImageOperation):
    def __init__(self, input, output, **kwargs):
        super().__init__(input, output, **kwargs)
        self.name = 'image_calculator'
        self.metadata = json.load(open(os.path.join(self.input, f'.{settings.INFO_FILE_NAME}'), 'r'))
        self.sequence = ",".join([self.metadata.get("sequence", ""), self.name])
        self.channel = self.metadata['channel']
        self.resolution_level = self.metadata['resolution_level']
        self.user = kwargs.get('user', get_user(self.input))
        self.priority = kwargs.get('priority', '2')
        self.input2 = kwargs['input2']
        self.calculator_operation = kwargs.get('calculator_operation', 'add')
        self.save_as_float = kwargs.get('save_as_float', False)
        self.scalar_operand = parse_scalar_operand(self.input2)
        self.prerequisites = kwargs.get('prerequisites', [])
        self.z_selection = normalize_z_range(
            self.metadata,
            kwargs.get('z_start', 0),
            kwargs.get('z_end', -1)
        )
        self.z_start = self.z_selection['start']
        self.z_end = self.z_selection['end']
        self.z_suffix = z_range_suffix(self.z_selection)

        output_operation_folder = os.path.join(self.output, self.name)
        output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, f"slurm_jobs{self.z_suffix}")
        save_folder_name = f"image_calculator_{self.calculator_operation}"
        if self.save_as_float:
            save_folder_name += "_float"
        save_folder_name += self.z_suffix
        self.save_folder = os.path.join(output_folder_sequence, save_folder_name)
        print("Image calculator prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.file_pairs = self.create_file_pairs()
        first_input_path = next(iter(self.file_pairs.values()))['input1']
        if os.path.exists(first_input_path):
            self.input_dtype = str(tifffile.imread(first_input_path).dtype)
        else:
            self.input_dtype = self.metadata.get('dtype', 'same_as_input')
        self.output_dtype = 'float32' if self.save_as_float else self.input_dtype
        self.manifest_path = os.path.join(self.jobs_folder, 'file_pairs.json')

        if self.scalar_operand is not None and self.calculator_operation in {'and', 'or', 'xor'}:
            raise ValueError(f"Scalar second operand is not supported for '{self.calculator_operation}'")

    def create_file_pairs(self):
        files1 = index_tiff_files(
            self.input,
            self.metadata,
            allow_expected=(
                bool(self.prerequisites)
                and 'processed_z_range' in self.metadata
            )
        )
        selected_indices = range(self.z_start, self.z_end)
        missing_input1 = [z for z in selected_indices if z not in files1]
        if missing_input1:
            raise FileNotFoundError(
                f"First operand is missing selected z indices: {missing_input1[:10]}"
            )

        if self.calculator_operation == 'not':
            return {
                str(z): {
                    'input1': files1[z],
                    'input2': None,
                    'input2_scalar': None,
                    'output': os.path.join(self.save_folder, os.path.basename(files1[z])),
                }
                for z in selected_indices
            }

        if self.scalar_operand is not None:
            return {
                str(z): {
                    'input1': files1[z],
                    'input2': None,
                    'input2_scalar': self.scalar_operand,
                    'output': os.path.join(self.save_folder, os.path.basename(files1[z])),
                }
                for z in selected_indices
            }

        metadata2_path = os.path.join(self.input2, f'.{settings.INFO_FILE_NAME}')
        metadata2 = self.metadata
        if os.path.exists(metadata2_path):
            with open(metadata2_path, 'r') as f:
                metadata2 = json.load(f)
        files2 = index_tiff_files(
            self.input2,
            metadata2,
            allow_expected=False
        )
        if not files2:
            raise FileNotFoundError(f"No TIFF files found in second operand folder: {self.input2}")

        file_pairs = {}
        files2_by_name = {
            os.path.basename(path): path for path in files2.values()
        }
        missing = []
        for z in selected_indices:
            input1_path = files1[z]
            basename = os.path.basename(input1_path)
            if Z_FILENAME_PATTERN.search(basename):
                input2_path = files2.get(z)
            else:
                input2_path = files2_by_name.get(basename)
            if input2_path is None:
                missing.append(z)
                continue
            file_pairs[str(z)] = {
                'input1': input1_path,
                'input2': input2_path,
                'input2_scalar': None,
                'output': os.path.join(self.save_folder, basename),
            }
        if missing:
            raise FileNotFoundError(
                f"Second operand is missing selected z indices: {missing[:10]}"
            )
        return file_pairs

    def write_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.file_pairs, f)

    def run(self):
        print("Running image calculator")
        self.write_manifest()
        provenance_file_path = self.create_provenance()
        job_ids = self.do_image_calculation()
        return provenance_file_path, job_ids

    def create_provenance(self):
        source = os.path.join(self.input, f'.{settings.INFO_FILE_NAME}')
        base_output_dir = self.metadata['base_output_dir']
        base_input_dir = self.metadata['base_input_dir']
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
                    "second_operand": self.input2,
                    "second_operand_type": "scalar" if self.scalar_operand is not None else "folder",
                    "calculator_operation": self.calculator_operation,
                    "save_as_float": self.save_as_float,
                    "output_dtype": self.output_dtype,
                    "z_start": self.z_selection['requested_start'],
                    "z_end": self.z_selection['requested_end'],
                }
            },
            "source": source,
            "channel": self.channel,
            "resolution_level": self.resolution_level,
            "resolution": self.metadata['resolution'],
            "shape": self.metadata['shape'],
            "orientation": self.metadata['orientation'],
            "base_output_dir": base_output_dir,
            "base_input_dir": base_input_dir,
            "sequence": self.sequence,
            "processed_z_range": z_range_provenance(self.z_selection)
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def do_image_calculation(self):
        path_to_task = os.path.join(self.jobs_folder, "image_calculator.sh")
        main_script = os.path.abspath(__file__)
        slurm_script = os.path.join(os.path.dirname(main_script), "do_image_calculation.py")
        with open(path_to_task, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write('\n')
            f.write(f"#SBATCH -J {self.user}-image-calculator")
            f.write('\n')
            f.write(f"#SBATCH -o {self.jobs_folder}/slurm_%j.out")
            f.write('\n')
            f.write('\n')
            f.write("source /h20/home/lab/miniconda3/bin/activate peace")
            f.write('\n')
            f.write(f'python {slurm_script}')
            f.write(' ')
            f.write(self.manifest_path if ' ' not in self.manifest_path else f'"{self.manifest_path}"')
            f.write(' ')
            f.write('$SLURM_ARRAY_TASK_ID')
            f.write(' ')
            f.write(self.calculator_operation)
            f.write(' ')
            f.write(str(self.save_as_float))
            f.write('\n')

        extra_args = {}
        if self.prerequisites:
            extra_args['--depend'] = f'afterok:{":".join(list(map(str, self.prerequisites)))}'
            extra_args['--kill-on-invalid-dep'] = 'yes'

        task_indices = sorted(int(z) for z in self.file_pairs)
        existing_outputs = {
            int(z) for z, pair in self.file_pairs.items()
            if os.path.exists(pair['output'])
        }
        return submit_slurm_indices(
            path_to_task,
            task_indices,
            existing_outputs=existing_outputs,
            partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
            cores=1,
            memory=32,
            priority=self.priority,
            extra_args=extra_args
        )
