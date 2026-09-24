import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def fake_run(stdout="", returncode=0, counter=None):
    """A subprocess.run stand-in that records calls; optionally increments a
    counter so each call can return a distinct job number."""
    calls = []

    def _run(command, *args, **kwargs):
        calls.append(list(command))
        if counter is not None:
            counter.append(len(calls))
            return _FakeCompleted(stdout=f"Submitted batch job {10000 + len(calls)}\n")
        return _FakeCompleted(stdout=stdout, stderr="", returncode=returncode)
    _run.calls = calls
    return _run


@pytest.fixture(scope="session")
def validation():
    from utils import validation
    return validation
