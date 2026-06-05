import json
import os
from glob import glob

import tifffile

from ..base import ImageOperation
from analysis import settings
from utils import get_user
from utils.containers import build_container_exec_prefix
from utils.slurm import split_slurm_array, submit_slurm_array


def get_tiff_files(folder):
    files = glob(os.path.join(folder, "*.tif"))
    files.extend(glob(os.path.join(folder, "*.tiff")))
    return sorted(files)


def parse_scalar_operand(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

        output_operation_folder = os.path.join(self.output, self.name)
        output_folder_sequence = os.path.join(
            output_operation_folder,
            f"resolution_level_{self.resolution_level}",
            f"channel_{self.channel}",
            f"{self.sequence}"
        )
        self.jobs_folder = os.path.join(output_folder_sequence, "slurm_jobs")
        save_folder_name = f"image_calculator_{self.calculator_operation}"
        if self.save_as_float:
            save_folder_name += "_float"
        self.save_folder = os.path.join(output_folder_sequence, save_folder_name)
        self.prerequisites = kwargs.get('prerequisites', [])
        print("Image calculator prerequisites", self.prerequisites)
        os.umask(settings.UMASK)
        if not os.path.exists(self.jobs_folder):
            os.makedirs(self.jobs_folder)
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.file_pairs = self.create_file_pairs()
        self.input_dtype = str(tifffile.imread(self.file_pairs[0]['input1']).dtype)
        self.output_dtype = 'float32' if self.save_as_float else self.input_dtype
        self.manifest_path = os.path.join(self.jobs_folder, 'file_pairs.json')

        if self.scalar_operand is not None and self.calculator_operation in {'and', 'or', 'xor'}:
            raise ValueError(f"Scalar second operand is not supported for '{self.calculator_operation}'")

    def create_file_pairs(self):
        files1 = get_tiff_files(self.input)

        if not files1:
            raise FileNotFoundError(f"No TIFF files found in first operand folder: {self.input}")

        if self.calculator_operation == 'not':
            return [
                {
                    'input1': input1_path,
                    'input2': None,
                    'input2_scalar': None,
                    'output': os.path.join(self.save_folder, os.path.basename(input1_path)),
                }
                for input1_path in files1
            ]

        if self.scalar_operand is not None:
            return [
                {
                    'input1': input1_path,
                    'input2': None,
                    'input2_scalar': self.scalar_operand,
                    'output': os.path.join(self.save_folder, os.path.basename(input1_path)),
                }
                for input1_path in files1
            ]

        files2 = get_tiff_files(self.input2)
        if not files2:
            raise FileNotFoundError(f"No TIFF files found in second operand folder: {self.input2}")

        files2_by_name = {os.path.basename(path): path for path in files2}
        file_pairs = []
        missing = []

        for input1_path in files1:
            basename = os.path.basename(input1_path)
            input2_path = files2_by_name.get(basename)
            if self.calculator_operation != 'not' and input2_path is None:
                missing.append(basename)
                continue
            file_pairs.append({
                'input1': input1_path,
                'input2': input2_path,
                'input2_scalar': None,
                'output': os.path.join(self.save_folder, basename),
            })

        if missing:
            missing_preview = ", ".join(missing[:10])
            raise FileNotFoundError(
                f"Missing matching TIFF files in second operand folder for: {missing_preview}"
            )
        if not file_pairs:
            raise ValueError("No matching TIFF pairs found to process")
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
            "sequence": self.sequence
        }
        provenance_file_path = os.path.join(self.save_folder, f'.{settings.INFO_FILE_NAME}')
        with open(provenance_file_path, "w") as f:
            f.write(json.dumps(provenance))
        return provenance_file_path

    def do_image_calculation(self):
        number_of_tasks = len(self.file_pairs)
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
            f.write(build_container_exec_prefix('peace', os.path.dirname(main_script)))
            f.write(f' python {slurm_script}')
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

        processed_names = {
            os.path.basename(path) for path in get_tiff_files(self.save_folder)
        }
        if processed_names:
            existing_outputs = [
                idx for idx, pair in enumerate(self.file_pairs)
                if os.path.basename(pair['output']) in processed_names
            ]
            print("Partially processed")
            print("Processed", len(existing_outputs), "of", number_of_tasks)
            job_ids = split_slurm_array(
                path_to_task,
                number_of_tasks,
                existing_outputs,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=1,
                memory=32,
                priority=self.priority,
                extra_args=extra_args
            )
        else:
            job_ids = submit_slurm_array(
                path_to_task,
                number_of_tasks,
                partition=','.join([settings.SLURM_PARTITION_CPU, settings.SLURM_PARTITION_HIGH_RAM]),
                cores=1,
                memory=32,
                priority=self.priority,
                extra_args=extra_args
            )
        return job_ids
