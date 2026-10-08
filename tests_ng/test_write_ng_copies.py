"""Tests for operations/brainreg/write_ng_copies.py — the BrAinPI-friendly
_ng.tif OME copies the brainreg operation writes after registration.

brainreg's plain 3D tiffs report axes 'QYX', which BrAinPI's tiff loader
rejects (TCZYXS only). The copies must report 'ZYX', preserve shape/dtype and
physical pixel size, and be discoverable by the CLI's folder search.

Lives in tests_ng/ because the lab-owned tests/ directory is not
group-writable; pytest.ini collects both directories.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

OPS_BRAINREG = Path(__file__).resolve().parents[1] / "operations" / "brainreg"
STANDARD_AXES = set("TCZYXS")


def load_module():
    """Load the helper by file path — importing through operations/__init__
    would pull in every heavy operation dependency."""
    spec = importlib.util.spec_from_file_location(
        "write_ng_copies", OPS_BRAINREG / "write_ng_copies.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load_module()


def make_plain_tiff(folder, name="downsampled.tiff", shape=(10, 64, 64), resolution=None):
    """Write a plain (non-OME) 3D tiff like brainreg does — tifffile reports
    its axes as 'QYX', which is outside BrAinPI's TCZYXS."""
    array = np.random.randint(0, 255, shape, dtype=np.uint8)
    src = str(Path(folder) / name)
    if resolution is not None:
        tifffile.imwrite(src, array, resolution=resolution)
    else:
        tifffile.imwrite(src, array)
    return src, array


def test_source_is_rejected_copy_is_accepted(tmp_path, mod):
    """The whole point: the plain source fails BrAinPI's axes check, the _ng
    copy passes it."""
    src, _ = make_plain_tiff(tmp_path)
    with tifffile.TiffFile(src) as tif:
        source_axes = tif.series[0].axes
    assert not set(source_axes).issubset(STANDARD_AXES), (
        f"expected a BrAinPI-incompatible source, got axes '{source_axes}'"
    )

    dst = mod.write_ng_copy(src)
    assert dst == str(Path(tmp_path) / "downsampled_ng.tif")
    assert os.path.isfile(dst)
    with tifffile.TiffFile(dst) as tif:
        assert tif.series[0].axes == "ZYX"
        assert set(tif.series[0].axes).issubset(STANDARD_AXES)
        assert tif.series[0].shape == (10, 64, 64)
        assert str(tif.series[0].dtype) == "uint8"


def test_copy_is_pixel_exact(tmp_path, mod):
    src, array = make_plain_tiff(tmp_path)
    dst = mod.write_ng_copy(src)
    assert np.array_equal(tifffile.imread(dst), array)


def test_resolution_preserved(tmp_path, mod):
    """Physical pixel size (XResolution/YResolution) survives the rewrite."""
    src, _ = make_plain_tiff(tmp_path, resolution=(1.0, 1.0))
    dst = mod.write_ng_copy(src)
    with tifffile.TiffFile(dst) as tif:
        xres = tif.pages[0].tags["XResolution"].value
    assert abs(xres[0] / xres[1] - 1.0) < 1e-6


def test_skip_existing_and_overwrite(tmp_path, mod):
    src, _ = make_plain_tiff(tmp_path)
    first = mod.write_ng_copy(src)
    assert first is not None
    assert mod.write_ng_copy(src) is None, "existing copy must be skipped"
    assert mod.write_ng_copy(src, overwrite=True) is not None


def test_unsupported_ndim_raises(tmp_path, mod):
    """2-D images map to 'YX' (accepted by BrAinPI); 4-D and up are rejected."""
    src = str(Path(tmp_path) / "flat.tif")
    tifffile.imwrite(src, np.random.randint(0, 255, (64, 64), dtype=np.uint8))
    dst = mod.write_ng_copy(src)
    assert dst is not None
    with tifffile.TiffFile(dst) as tif:
        assert tif.series[0].axes == "YX"

    four_d = str(Path(tmp_path) / "stack.tif")
    tifffile.imwrite(four_d, np.random.randint(0, 255, (2, 10, 64, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match="4-D image"):
        mod.write_ng_copy(four_d)


def test_collect_output_folders_direct_and_recursive(tmp_path, mod):
    direct = tmp_path / "reg_a"
    direct.mkdir()
    make_plain_tiff(direct)
    make_plain_tiff(direct, name="boundaries.tiff")

    nested = tmp_path / "parent" / "deep" / "reg_b"
    nested.mkdir(parents=True)
    make_plain_tiff(nested)
    make_plain_tiff(nested, name="boundaries.tiff")

    partial = tmp_path / "partial"
    partial.mkdir()
    make_plain_tiff(partial)  # downsampled.tiff only — must not match

    found = list(mod.collect_output_folders([str(direct), str(tmp_path / "parent")]))
    assert str(direct) in found
    assert str(nested) in found
    assert str(partial) not in found


def test_cli_main(tmp_path, mod, capsys):
    folder = tmp_path / "reg"
    folder.mkdir()
    make_plain_tiff(folder)
    make_plain_tiff(folder, name="boundaries.tiff")

    assert mod.main([str(folder)]) == 0
    assert (folder / "downsampled_ng.tif").is_file()
    assert (folder / "boundaries_ng.tif").is_file()
    out = capsys.readouterr().out
    assert "Done: 2 written" in out

    # re-run without --overwrite skips; with --overwrite rewrites
    assert mod.main([str(folder)]) == 0
    assert "Done: 0 written, 2 skipped" in capsys.readouterr().out
    assert mod.main([str(folder), "--overwrite"]) == 0
    assert "Done: 2 written" in capsys.readouterr().out


def test_cli_main_no_folders(tmp_path, mod, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mod.main([str(empty)]) == 1
    assert "No brainreg output folders found" in capsys.readouterr().out
