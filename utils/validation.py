"""Security validation helpers for the pipeline watcher.

Import-light on purpose (no numpy/pandas) so the test suite can exercise them
without the analysis environment.
"""

import os
import re

_RECORD_ID_PATTERN = re.compile(r'[0-9a-fA-F-]+')


def valid_user(user):
    """Lab usernames only: no whitespace/newlines/shell metacharacters."""
    return isinstance(user, str) and bool(re.fullmatch(r'[A-Za-z0-9_.-]+', user))


def path_allowed_for_user(path, user, fs_root):
    """Realpath containment: the path must resolve inside {fs_root}/{user}/
    (symlinks included). Prevents '..' traversal and prefix collisions such as
    /h20/Public/iana-evil passing a check for /h20/Public/iana."""
    if not path or not valid_user(user):
        return False
    root = os.path.realpath(os.path.join(fs_root, user))
    real = os.path.realpath(path)
    return real == root or real.startswith(root + os.sep)


def resolve_operation_class(name, namespace, plugins, reader_plugins, base_classes, allow_reader=False):
    """Only registered operation/reader classes may be dispatched."""
    cls = getattr(namespace, name, None)
    if isinstance(cls, type) and any(issubclass(cls, base) for base in base_classes):
        return cls
    if allow_reader and name in reader_plugins:
        return reader_plugins[name]
    if name in plugins:
        return plugins[name]
    raise ValueError(f"Unknown operation: {name}")


def history_record_path(payload, kind, history_dir):
    """Derive the history record path from job_record_id/workflow_id and the
    submitting user instead of trusting history_file from the task JSON
    (a crafted JSON must not be able to rewrite arbitrary .json files)."""
    record_id = payload.get('job_record_id') or payload.get('workflow_id')
    submitted_by = payload.get('submitted_by')
    if not record_id or not valid_user(submitted_by):
        return None
    if not re.fullmatch(r'[0-9a-fA-F-]+', str(record_id)):
        return None
    folder = 'workflows' if kind == 'workflow' else 'jobs'
    return os.path.join(history_dir, folder, submitted_by, f'{record_id}.json')
