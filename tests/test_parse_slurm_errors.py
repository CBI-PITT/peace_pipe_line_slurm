"""parse_slurm_errors: log triage with tmp directories."""

import pytest

from utils.slurm import parse_slurm_errors


def test_moves_error_files_to_errors_folder(tmp_path):
    logs = tmp_path / "logs"
    (logs / "sub").mkdir(parents=True)
    (logs / "clean.out").write_text("all good\n")
    (logs / "bad.out").write_text("Traceback (most recent call last)\nValueError: boom\n")
    (logs / "sub" / "bad2.out").write_text("Permission denied\n")
    (logs / "notes.txt").write_text("Traceback (most recent call last)\n")  # non-.out ignored

    flag = parse_slurm_errors(str(logs))
    assert flag is True
    errors = logs / "errors"
    assert (logs / "clean.out").exists(), "clean logs must stay put"
    assert not (logs / "bad.out").exists()
    assert (errors / "bad.out").exists()
    assert (errors / "sub__bad2.out").exists(), "nested logs flatten into errors/"
    assert not (errors / "notes.txt").exists(), "non-.out files are never moved"


def test_error_patterns_are_matched_case_insensitively(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    for name, text in [("a.out", "Segmentation fault\n"), ("b.out", "MODULENOTFOUNDERROR\n"),
                       ("c.out", "command not found\n"), ("d.out", "FileNotFoundError\n")]:
        (logs / name).write_text(text)
    assert parse_slurm_errors(str(logs)) is True
    assert (logs / "errors" / "a.out").exists()
    assert (logs / "errors" / "b.out").exists()
    assert (logs / "errors" / "c.out").exists()
    assert (logs / "errors" / "d.out").exists()


def test_no_errors_returns_false(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "ok.out").write_text("processed 100 planes\n")
    assert parse_slurm_errors(str(logs)) is False
    assert (logs / "ok.out").exists()


def test_idempotent_second_run(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "bad.out").write_text("Error: something\n")
    assert parse_slurm_errors(str(logs)) is True
    assert parse_slurm_errors(str(logs)) is False, (
        "files already in errors/ must be skipped on re-processing"
    )
    assert (logs / "errors" / "bad.out").exists()


def test_duplicate_basenames_do_not_clobber(tmp_path):
    logs = tmp_path / "logs"
    (logs / "run1").mkdir(parents=True)
    (logs / "run2").mkdir(parents=True)
    (logs / "run1" / "job.out").write_text("Error: first\n")
    (logs / "run2" / "job.out").write_text("Error: second\n")
    parse_slurm_errors(str(logs))
    errors = logs / "errors"
    assert (errors / "run1__job.out").exists()
    assert (errors / "run2__job.out").exists()
    assert (errors / "run1__job.out").read_text() == "Error: first\n"


def test_missing_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        parse_slurm_errors(str(tmp_path / "missing"))
