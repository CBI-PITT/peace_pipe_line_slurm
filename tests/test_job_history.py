"""Backend utils/job_history.py: record load/update/failure-marking."""

import json
import os

import pytest

from analysis import settings
from utils import job_history


def _write_record(tmp_path, payload):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _job_record(**overrides):
    record = {
        "kind": "job",
        "job_record_id": "abc123",
        "submitted_by": "iana",
        "status": "submitted",
        "slurm_job_ids": [],
        "steps": [],
    }
    record.update(overrides)
    return record


def test_history_enabled():
    assert job_history.history_enabled() is True


def test_load_record(tmp_path):
    path = _write_record(tmp_path, _job_record())
    assert job_history.load_record(path)["job_record_id"] == "abc123"
    assert job_history.load_record(str(tmp_path / "missing.json")) is None
    assert job_history.load_record("") is None


def test_history_disabled_blocks_reads(monkeypatch, tmp_path):
    path = _write_record(tmp_path, _job_record())
    monkeypatch.setattr(settings, "ENABLE_JOB_HISTORY", False)
    assert job_history.load_record(path) is None
    assert job_history.update_record(path, {"status": "x"}) is None
    assert job_history.mark_dispatch_failure(path, "boom") is None
    monkeypatch.undo()


def test_update_record_merges_and_persists(tmp_path):
    path = _write_record(tmp_path, _job_record(slurm_job_ids=["7"]))
    updated = job_history.update_record(path, {"status": "cancelled"})
    assert updated["status"] == "cancelled"
    assert updated["slurm_job_ids"] == ["7"], "unrelated fields must survive the merge"
    reloaded = json.loads(open(path).read())
    assert reloaded["status"] == "cancelled"
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o664, "history records are written group-readable (chmod 664)"


def test_update_record_missing_path_returns_none(tmp_path):
    assert job_history.update_record(str(tmp_path / "missing.json"), {"status": "x"}) is None


def test_update_workflow_step(tmp_path):
    path = _write_record(tmp_path, {
        "kind": "workflow",
        "status": "submitted",
        "steps": [
            {"step_id": "s1", "status": "submitted"},
            {"step_id": "s2", "status": "submitted"},
        ],
    })
    updated = job_history.update_workflow_step(
        path, "s1", {"status": "finished successfully"}, workflow_updates={"note": "x"})
    assert updated["steps"][0]["status"] == "finished successfully"
    assert updated["steps"][1]["status"] == "submitted"
    assert updated["note"] == "x"
    assert json.loads(open(path).read())["steps"][0]["status"] == "finished successfully"


def test_mark_dispatch_failure_job(tmp_path):
    path = _write_record(tmp_path, _job_record(status="dispatching"))
    updated = job_history.mark_dispatch_failure(path, "NameError: boom")
    assert updated["status"] == "failed_to_dispatch"
    assert updated["dispatch_error"] == "NameError: boom"


def test_mark_dispatch_failure_workflow_updates_steps(tmp_path):
    path = _write_record(tmp_path, {
        "kind": "workflow",
        "status": "dispatching",
        "steps": [
            {"step_id": "s1", "status": "submitted"},
            {"step_id": "s2", "status": "finished successfully"},
        ],
    })
    updated = job_history.mark_dispatch_failure(path, "no job number")
    assert updated["status"] == "failed_to_dispatch"
    assert updated["steps"][0]["status"] == "failed_to_dispatch"
    assert updated["steps"][1]["status"] == "finished successfully", (
        "already-finished steps must keep their status"
    )


def test_infer_log_folder_from_operation(tmp_path):
    op = type("Op", (), {"jobs_folder": "/tmp/jobs"})()
    assert job_history.infer_log_folder(op) == "/tmp/jobs"


def test_infer_log_folder_walks_up_to_slurm_jobs(tmp_path):
    prov_dir = tmp_path / "exp" / "resolution_level_0" / "channel_1"
    prov_dir.mkdir(parents=True)
    slurm_jobs = tmp_path / "exp" / "slurm_jobs"
    slurm_jobs.mkdir()
    op = type("Op", (), {"jobs_folder": None})()
    provenance = str(prov_dir / "prov.json")
    assert job_history.infer_log_folder(op, provenance) == str(slurm_jobs)


def test_infer_log_folder_none_without_slurm_jobs(tmp_path):
    prov_dir = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"
    prov_dir.mkdir(parents=True)
    op = type("Op", (), {"jobs_folder": None})()
    assert job_history.infer_log_folder(op, str(prov_dir / "prov.json")) is None
    assert job_history.infer_log_folder(op) is None
