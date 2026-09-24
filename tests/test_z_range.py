"""utils/z_range.py: range normalization, suffixes, and TIFF path resolution."""

import pytest

from utils.z_range import (
    existing_z_indices,
    get_available_z_range,
    normalize_z_range,
    resolve_tiff_path,
    z_range_provenance,
    z_range_suffix,
    Z_FILENAME_PATTERN,
    Z_INDEX_PATTERN,
)

META = {"shape": [10, 512, 512]}  # z=10, y=512, x=512


def test_normalize_default_full_range():
    sel = normalize_z_range(META)
    assert sel["start"] == 0 and sel["end"] == 10
    assert sel["total_z"] == 10 and sel["is_full"] is True


def test_normalize_explicit_range():
    sel = normalize_z_range(META, 2, 7)
    assert sel["start"] == 2 and sel["end"] == 7
    assert sel["is_full"] is False


def test_normalize_minus_one_means_total():
    sel = normalize_z_range(META, 0, -1)
    assert sel["end"] == 10


@pytest.mark.parametrize("z_start, z_end", [
    (True, 5), (0, False),          # bools are not integers here
    (1.5, 8), (0, 7.5),             # non-integer floats
    ("abc", 8), (0, None),          # non-numeric
    (-1, 8),                        # negative start
    (0, -2),                        # end below -1
    (5, 5), (10, -1),               # start >= end
    (0, 11),                        # end beyond the source depth
])
def test_normalize_rejects_invalid(z_start, z_end):
    with pytest.raises(ValueError):
        normalize_z_range(META, z_start, z_end)


def test_normalize_accepts_integer_floats_and_numeric_strings():
    assert normalize_z_range(META, 2.0, 7.0)["end"] == 7
    assert normalize_z_range(META, "3", "8")["start"] == 3


def test_normalize_respects_processed_z_range():
    meta = {**META, "processed_z_range": {"start": 3, "end": 8}}
    sel = normalize_z_range(meta, 3, 8)
    assert sel["start"] == 3 and sel["end"] == 8
    with pytest.raises(ValueError):
        normalize_z_range(meta, 2, 7), "2 is below the available start"
    with pytest.raises(ValueError):
        normalize_z_range(meta, 0, -1), "-1 resolves to 10, beyond the available end"


def test_get_available_z_range():
    assert get_available_z_range(META) == (0, 10)
    meta = {**META, "processed_z_range": {"start": 3, "end": 8}}
    assert get_available_z_range(meta) == (3, 8)


def test_z_range_suffix_and_provenance():
    assert z_range_suffix(normalize_z_range(META)) == ""
    assert z_range_suffix(normalize_z_range(META, 2, 7)) == "_z2-7"
    prov = z_range_provenance(normalize_z_range(META, 2, 7))
    assert prov == {"start": 2, "end": 7, "end_exclusive": True}


def test_z_index_patterns():
    assert Z_FILENAME_PATTERN.search("stack_z12.tif").group(1) == "12"
    assert Z_FILENAME_PATTERN.search("stack_Z5.TIFF").group(1) == "5"
    assert Z_FILENAME_PATTERN.search("stack_z3.info") is None
    assert Z_INDEX_PATTERN.search("stack_z3.info").group(1) == "3"


def test_existing_z_indices(tmp_path):
    folder = tmp_path / "stack"
    folder.mkdir()
    for name in ("stack_z0.tif", "stack_z3.tif", "stack_z7.TIFF", "notes.txt", "stack_z9.info"):
        (folder / name).write_bytes(b"x")
    assert existing_z_indices(str(folder)) == {0, 3}, "tif only by default"
    assert existing_z_indices(str(folder), extension="tiff") == {7}
    assert existing_z_indices(str(folder), extension="info") == {9}
    assert existing_z_indices(str(tmp_path / "missing")) == set()


def test_resolve_tiff_path_canonical_name(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    canonical = folder / "r00_t00_c01_z0005.tif"
    canonical.write_bytes(b"x")
    assert resolve_tiff_path(str(folder), META, 0, 1, 5) == str(canonical)


def test_resolve_tiff_path_indexed_names(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / "a_z2.tif").write_bytes(b"x")
    (folder / "a_z9.tif").write_bytes(b"x")
    assert resolve_tiff_path(str(folder), META, 0, 0, 9).endswith("a_z9.tif")
    with pytest.raises(FileNotFoundError):
        resolve_tiff_path(str(folder), META, 0, 0, 5)


def test_resolve_tiff_path_rejects_mixed_names(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / "a_z2.tif").write_bytes(b"x")
    (folder / "plain.tif").write_bytes(b"x")
    with pytest.raises(ValueError):
        resolve_tiff_path(str(folder), META, 0, 0, 2)


def test_resolve_tiff_path_unindexed_uses_metadata(tmp_path):
    folder = tmp_path / "r"
    folder.mkdir()
    (folder / "b.tif").write_bytes(b"x")
    (folder / "a.tif").write_bytes(b"x")
    meta = {"shape": [2, 512, 512]}  # exactly as many files as the available range
    assert resolve_tiff_path(str(folder), meta, 0, 0, 0).endswith("a.tif")
    assert resolve_tiff_path(str(folder), meta, 0, 0, 1).endswith("b.tif")
    with pytest.raises(ValueError):
        resolve_tiff_path(str(folder), META, 0, 0, 0), "count mismatch vs the 10-deep range"
    with pytest.raises(FileNotFoundError):
        empty = tmp_path / "empty"
        empty.mkdir()
        resolve_tiff_path(str(empty), None, 0, 0, 0)
