import json
import os
import re
import shlex

from analysis import settings


SHELL_VARIABLE_PATTERN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def get_container_runtime():
    runtime = settings.CONTAINER_RUNTIME
    if runtime not in ['apptainer', 'singularity']:
        raise ValueError(
            f"Unsupported container runtime '{runtime}'. "
            "Expected 'apptainer' or 'singularity'."
        )
    return runtime


def load_container_manifest(base_dir):
    manifest_path = os.path.join(base_dir, 'containers.json')
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    containers = manifest.get('containers', {})
    if type(containers) is not dict:
        raise ValueError(f"Field 'containers' in {manifest_path} must be a dictionary")
    return manifest


def get_fallback_container_dir():
    fallback_dir = settings.CONTAINER_FALLBACK_DIR
    if fallback_dir is None:
        return None
    return str(fallback_dir)


def resolve_container_path(container_name, base_dir):
    manifest = load_container_manifest(base_dir)
    checked_paths = []
    manifest_container = manifest.get('containers', {}).get(container_name)

    if manifest_container is not None:
        if os.path.isabs(manifest_container):
            checked_paths.append(manifest_container)
        else:
            checked_paths.append(os.path.join(base_dir, manifest_container))
    else:
        checked_paths.append(os.path.join(base_dir, f'{container_name}.sif'))

    fallback_dir = get_fallback_container_dir()
    if fallback_dir is not None:
        checked_paths.append(os.path.join(fallback_dir, f'{container_name}.sif'))

    for container_path in checked_paths:
        if os.path.exists(container_path):
            return container_path

    checked_paths_text = '\n'.join(f'- {path}' for path in checked_paths)
    raise FileNotFoundError(
        f"Container '{container_name}' was not found. Checked:\n{checked_paths_text}"
    )


def quote_shell_arg(value):
    value = str(value)
    if SHELL_VARIABLE_PATTERN.fullmatch(value):
        return value
    return shlex.quote(value)


def build_container_exec_prefix(container_name, base_dir, needs_gpu=False):
    runtime = get_container_runtime()
    container_path = resolve_container_path(container_name, base_dir)
    command_parts = [runtime, 'exec']
    if needs_gpu:
        command_parts.append('--nv')
    command_parts.append(container_path)
    return ' '.join(quote_shell_arg(part) for part in command_parts)


def build_container_command(container_name, base_dir, command_parts, needs_gpu=False):
    prefix = build_container_exec_prefix(container_name, base_dir, needs_gpu=needs_gpu)
    command = ' '.join(quote_shell_arg(part) for part in command_parts)
    return f'{prefix} {command}'


def build_container_python_command(container_name, base_dir, script_path, script_args=None, needs_gpu=False):
    if script_args is None:
        script_args = []
    command_parts = ['python', script_path]
    command_parts.extend(script_args)
    return build_container_command(
        container_name,
        base_dir,
        command_parts,
        needs_gpu=needs_gpu
    )
