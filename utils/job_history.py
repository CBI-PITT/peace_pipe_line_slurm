import copy
import json
import os
import tempfile

from analysis import settings


def history_enabled():
    return getattr(settings, 'ENABLE_JOB_HISTORY', False)


def _write_json(path, payload):
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_history_', dir=parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
        os.chmod(path, 0o664)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_record(path):
    if not history_enabled() or not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def update_record(path, updates):
    record = load_record(path)
    if record is None:
        return None
    record.update(copy.deepcopy(updates))
    _write_json(path, record)
    return record


def update_workflow_step(path, step_id, updates, workflow_updates=None):
    record = load_record(path)
    if record is None:
        return None
    for step in record.get('steps', []):
        if step.get('step_id') == step_id:
            step.update(copy.deepcopy(updates))
            break
    if workflow_updates:
        record.update(copy.deepcopy(workflow_updates))
    _write_json(path, record)
    return record


def mark_dispatch_failure(path, error_text):
    record = load_record(path)
    if record is None:
        return None
    updates = {
        'status': 'failed_to_dispatch',
        'dispatch_error': error_text,
    }
    if record.get('kind') == 'workflow':
        for step in record.get('steps', []):
            if step.get('status') in ('submitted', 'dispatching'):
                step['status'] = 'failed_to_dispatch'
                step['dispatch_error'] = error_text
        record.update(updates)
        _write_json(path, record)
        return record
    return update_record(path, updates)


def infer_log_folder(operation, provenance_path=None):
    jobs_folder = getattr(operation, 'jobs_folder', None)
    if jobs_folder:
        return jobs_folder
    if not provenance_path:
        return None
    current = os.path.dirname(provenance_path)
    for _ in range(5):
        candidate = os.path.join(current, 'slurm_jobs')
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None
