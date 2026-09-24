"""Submission helpers in utils/slurm.py with mocked subprocess.run."""

import subprocess

import pytest

from conftest import fake_run
from utils.slurm import (
    get_job_number_from_slurm_out,
    split_indices_in_subranges,
    split_range_in_subranges,
    submit_partial_slurm_array,
    submit_slurm_array,
    submit_slurm_indices,
    submit_slurm_job,
)

SBATCH_STDOUT = "Submitted batch job 12345\n"


def test_get_job_number_from_slurm_out():
    out = type("R", (), {"stdout": SBATCH_STDOUT})()
    assert get_job_number_from_slurm_out(out) == 12345


def test_submit_slurm_job_command(monkeypatch):
    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    ids = submit_slurm_job("/tmp/job.sh", partition="compute", cores=4, memory=16)
    assert ids == [12345]
    cmd = calls.calls[0]
    assert cmd[0] == "sbatch"
    assert "compute" in cmd
    assert "--mem=16Gb" in cmd
    assert "-n4" in cmd
    assert "--gres=gpu:1" not in cmd
    assert cmd[-1] == "/tmp/job.sh"


def test_submit_slurm_job_gpu_gres(monkeypatch):
    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    submit_slurm_job("/tmp/job.sh", partition="gpu", needs_gpu=True)
    assert "--gres=gpu:1" in calls.calls[0]

    calls2 = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls2)
    submit_slurm_job("/tmp/job.sh", partition="compute", needs_gpu=True)
    assert "--gres=gpu:1" not in calls2.calls[0], (
        "gpu reservation must only be requested for GPU-enabled partitions"
    )


def test_submit_slurm_job_nice_scaling(monkeypatch):
    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    submit_slurm_job("/tmp/job.sh", priority="0", cores=1)
    assert "--nice=41666" in calls.calls[0], "compute nice scales with cores (int(1000000*1/24))"

    calls = fake_run(SBATCH_STDOUT)
    submit_slurm_job("/tmp/job.sh", priority="0", cores=24)
    monkeypatch.setattr(subprocess, "run", calls)
    submit_slurm_job("/tmp/job.sh", priority="0", cores=24)
    assert "--nice=1000000" in calls.calls[0]

    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    submit_slurm_job("/tmp/job.sh", priority="0", cores=1, needs_gpu=True, partition="gpu")
    assert "--nice=1000000" in calls.calls[0], "gpu nice is not scaled by cores"


def test_submit_slurm_job_extra_args(monkeypatch):
    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    submit_slurm_job("/tmp/job.sh", extra_args={
        "--depend": "afterok:1:2", "--kill-on-invalid-dep": "yes"})
    cmd = calls.calls[0]
    assert "--depend=afterok:1:2" in cmd
    assert "--kill-on-invalid-dep=yes" in cmd
    assert cmd[-1] == "/tmp/job.sh", "the job script must be the last sbatch argument"


def test_submit_slurm_job_failure_returns_empty(monkeypatch):
    calls = fake_run("")  # no job number in sbatch output
    monkeypatch.setattr(subprocess, "run", calls)
    assert submit_slurm_job("/tmp/job.sh") == []

    def failing(*args, **kwargs):
        return type("R", (), {"stdout": "", "stderr": "boom", "returncode": 1})()
    monkeypatch.setattr(subprocess, "run", failing)
    assert submit_slurm_job("/tmp/job.sh") == []


def test_submit_slurm_array_range(monkeypatch):
    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    ids = submit_slurm_array("/tmp/job.sh", 5)
    assert ids == [12345]
    assert "--array=0-4" in calls.calls[0]


def test_submit_partial_slurm_array_range(monkeypatch):
    calls = fake_run(SBATCH_STDOUT)
    monkeypatch.setattr(subprocess, "run", calls)
    result = submit_partial_slurm_array("/tmp/job.sh", 2, 7)
    assert result == 12345, "partial submission returns the job number itself"
    assert "--array=2-7" in calls.calls[0]


@pytest.mark.parametrize("indices, expected", [
    ([0, 1, 2, 3], [(0, 3)]),
    ([0, 2, 5], [(0, 0), (2, 2), (5, 5)]),
    ([1, 2, 5, 6, 9], [(1, 2), (5, 6), (9, 9)]),
    ([], []),
    ([7], [(7, 7)]),
    ([3, 1, 2], [(1, 3)]),
])
def test_split_indices_in_subranges(indices, expected):
    assert split_indices_in_subranges(indices) == expected


def test_submit_slurm_indices_splits_and_skips_existing(monkeypatch):
    import re
    calls = []
    def counting_run(command, *a, **kw):
        calls.append(list(command))
        n = len(calls)
        return type("R", (), {"stdout": f"Submitted batch job {10000 + n}\n"})()
    monkeypatch.setattr(subprocess, "run", counting_run)

    job_ids = submit_slurm_indices("/tmp/job.sh", [0, 1, 2, 3, 4], existing_outputs=[1, 2])
    assert len(job_ids) == 2, "one array job per contiguous missing subrange"
    arrays = [re.search(r"--array=([\d-]+)", " ".join(c)).group(1) for c in calls]
    assert arrays == ["0-0", "3-4"], "existing z indices must be skipped"


def test_submit_slurm_indices_raises_on_failure(monkeypatch):
    calls = fake_run("")  # sbatch produced no job number
    monkeypatch.setattr(subprocess, "run", calls)
    with pytest.raises(RuntimeError):
        submit_slurm_indices("/tmp/job.sh", [0, 1, 2])


def test_split_range_in_subranges():
    assert split_range_in_subranges(5, [1, 2]) == [(0, 0), (3, 4)]
