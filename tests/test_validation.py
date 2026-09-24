"""Regression guards for the watcher's security validation (2026-09-23).

The audit found that USER came from the same attacker-controlled JSON as the
paths (self-referential check), 'startswith' allowed prefix collisions, and
history_file from the JSON could rewrite arbitrary .json files. These tests pin
the validation helpers; see utils/validation.py.
"""

import os
import types

import pytest


def test_valid_user_accepts_lab_names(validation):
    assert validation.valid_user("dutta-p")
    assert validation.valid_user("CBI_Admin")
    assert validation.valid_user("iana")
    assert validation.valid_user("user.name_1")


def test_valid_user_rejects_injection(validation):
    assert not validation.valid_user("iana\nrm -rf ~")
    assert not validation.valid_user("user;id")
    assert not validation.valid_user("user id")
    assert not validation.valid_user("$(reboot)")
    assert not validation.valid_user("")
    assert not validation.valid_user(None)
    assert not validation.valid_user(123)


def test_path_allowed_for_user_inside_root(validation, tmp_path):
    fs_root = tmp_path / "Public"
    user_dir = fs_root / "iana"
    user_dir.mkdir(parents=True)
    assert validation.path_allowed_for_user(str(user_dir / "data"), "iana", str(fs_root))
    assert validation.path_allowed_for_user(str(user_dir), "iana", str(fs_root))


def test_path_allowed_for_user_rejects_outside(validation, tmp_path):
    fs_root = tmp_path / "Public"
    user_dir = fs_root / "iana"
    user_dir.mkdir(parents=True)
    other = fs_root / "other-user"
    other.mkdir()
    assert not validation.path_allowed_for_user(str(other / "data"), "iana", str(fs_root))
    assert not validation.path_allowed_for_user("/etc/hostname", "iana", str(fs_root))
    assert not validation.path_allowed_for_user("", "iana", str(fs_root))
    assert not validation.path_allowed_for_user(str(user_dir / "x"), "nonexistent user", str(fs_root))


def test_path_allowed_for_user_rejects_traversal(validation, tmp_path):
    fs_root = tmp_path / "Public"
    user_dir = fs_root / "iana"
    other_dir = fs_root / "other-user"
    user_dir.mkdir(parents=True)
    other_dir.mkdir()
    traversal = str(user_dir / ".." / "other-user" / "data")
    assert not validation.path_allowed_for_user(traversal, "iana", str(fs_root))


def test_path_allowed_for_user_rejects_prefix_collision(validation, tmp_path):
    fs_root = tmp_path / "Public"
    (fs_root / "iana-evil").mkdir(parents=True)
    (fs_root / "iana").mkdir()
    assert not validation.path_allowed_for_user(str(fs_root / "iana-evil" / "x"), "iana", str(fs_root)), (
        "startswith without a trailing separator must not match sibling dirs"
    )


def test_path_allowed_for_user_rejects_symlink_escape(validation, tmp_path):
    fs_root = tmp_path / "Public"
    user_dir = fs_root / "iana"
    user_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, user_dir / "link")
    assert not validation.path_allowed_for_user(str(user_dir / "link" / "data"), "iana", str(fs_root))


class _FakeBase:
    pass


class _FakeOperation(_FakeBase):
    pass


class _FakeReader(_FakeBase):
    pass


def _namespace(**attrs):
    module = types.ModuleType("fake_module")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def test_resolve_operation_class_from_namespace(validation):
    namespace = _namespace(_FakeOperation=_FakeOperation, os=os)
    cls = validation.resolve_operation_class(
        "_FakeOperation", namespace, {}, {}, (_FakeBase,))
    assert cls is _FakeOperation


def test_resolve_operation_class_rejects_non_base_attributes(validation):
    """Attributes that are not registered operation classes must not dispatch."""
    namespace = _namespace(os=os)
    with pytest.raises(ValueError):
        validation.resolve_operation_class("os", namespace, {}, {}, (_FakeBase,))


def test_resolve_operation_class_from_registries(validation):
    plugins = {"custom_op": _FakeOperation}
    readers = {"reader": _FakeReader}
    namespace = _namespace()
    assert validation.resolve_operation_class(
        "custom_op", namespace, plugins, readers, (_FakeBase,)) is _FakeOperation
    assert validation.resolve_operation_class(
        "reader", namespace, plugins, readers, (_FakeBase,), allow_reader=True) is _FakeReader


def test_resolve_operation_class_unknown(validation):
    with pytest.raises(ValueError):
        validation.resolve_operation_class("nope", _namespace(), {}, {}, (_FakeBase,))


def test_history_record_path_job_and_workflow(validation, tmp_path):
    history_dir = str(tmp_path)
    job = validation.history_record_path(
        {"job_record_id": "abcd1234-0000", "submitted_by": "iana"}, "job", history_dir)
    assert job == os.path.join(history_dir, "jobs", "iana", "abcd1234-0000.json")
    wf = validation.history_record_path(
        {"workflow_id": "abcd1234-0000", "submitted_by": "iana"}, "workflow", history_dir)
    assert wf == os.path.join(history_dir, "workflows", "iana", "abcd1234-0000.json")


def test_history_record_path_rejects_crafted_payloads(validation, tmp_path):
    history_dir = str(tmp_path)
    # attacker-chosen history_file locations are ignored outright
    crafted = {"job_record_id": "..%2f..%2fetc", "submitted_by": "iana"}
    assert validation.history_record_path(crafted, "job", history_dir) is None
    assert validation.history_record_path(
        {"job_record_id": "x/../y", "submitted_by": "iana"}, "job", history_dir) is None
    assert validation.history_record_path(
        {"job_record_id": "abcd", "submitted_by": "bad user"}, "job", history_dir) is None
    assert validation.history_record_path(
        {"job_record_id": "abcd"}, "job", history_dir) is None
    assert validation.history_record_path({}, "job", history_dir) is None
