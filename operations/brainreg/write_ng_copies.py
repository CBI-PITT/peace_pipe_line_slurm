"""Write BrAinPI-friendly copies of brainreg's output tiffs.

brainreg writes downsampled.tiff / boundaries.tiff as plain 3D TIFFs; tifffile
reports their axes as 'QYX', which BrAinPI's tiff loader rejects (it only
allows TCZYXS). This script writes <name>_ng.tif copies next to the originals
as OME-TIFFs with declared 'ZYX' axes, which BrAinPI then serves to
Neuroglancer (the PEACE File Browser's "Neuroglancer" button for brainreg
outputs). The originals are left untouched.

Usage (manual, for existing outputs):
    python write_ng_copies.py <path> [<path> ...] [--overwrite]

Each path is either the brainreg output folder itself (the one containing
downsampled.tiff and boundaries.tiff) or any parent directory, which is
searched recursively for such folders. Existing *_ng.tif copies are skipped
unless --overwrite is given.

Requires tifffile (any env with brainreg/brainglobe installed).
"""
import os
import sys

NG_SUFFIX = "_ng.tif"
TIFF_BASENAMES = ("downsampled.tiff", "boundaries.tiff")
AXES_BY_NDIM = {2: "YX", 3: "ZYX"}


def write_ng_copy(src_path, overwrite=False):
    """
    Write a BrAinPI-compatible OME copy of a brainreg output tiff.

    Args:
        src_path (str): Path to downsampled.tiff or boundaries.tiff.
        overwrite (bool, optional): Rewrite an existing _ng copy. Defaults to False.

    Returns:
        str: Path to the written copy, or None when an existing copy was skipped.

    Raises:
        ValueError: When the source image has an unsupported number of dimensions.
    """
    import tifffile

    stem, _ = os.path.splitext(src_path)
    dst_path = stem + NG_SUFFIX
    if os.path.exists(dst_path) and not overwrite:
        return None

    with tifffile.TiffFile(src_path) as tif:
        array = tif.series[0].asarray()
        resolution = None
        try:
            xres = tif.pages[0].tags["XResolution"].value
            yres = tif.pages[0].tags["YResolution"].value
            resolution = (
                float(xres[0]) / float(xres[1]),
                float(yres[0]) / float(yres[1]),
            )
        except (KeyError, IndexError, TypeError, ZeroDivisionError):
            resolution = None

    axes = AXES_BY_NDIM.get(array.ndim)
    if axes is None:
        raise ValueError(f"Unsupported {array.ndim}-D image: {src_path}")

    metadata = {"axes": axes}
    if resolution is not None:
        tifffile.imwrite(
            dst_path, array, ome=True, metadata=metadata,
            resolution=resolution, photometric="minisblack",
        )
    else:
        tifffile.imwrite(
            dst_path, array, ome=True, metadata=metadata,
            photometric="minisblack",
        )
    return dst_path


def collect_output_folders(paths):
    """
    Yield brainreg output folders that hold the registration tiffs.

    Each given path is yielded itself when it contains the tiffs, otherwise
    it is searched recursively for folders that do.

    Args:
        paths (list): Paths to brainreg output folders or their parents.

    Yields:
        str: Folders containing downsampled.tiff and boundaries.tiff.
    """
    for start in paths:
        if os.path.isfile(os.path.join(start, TIFF_BASENAMES[0])):
            yield start
        else:
            for dirpath, _, filenames in os.walk(start):
                if all(name in filenames for name in TIFF_BASENAMES):
                    yield dirpath


def main(argv=None):
    """
    CLI entry point: write _ng copies for every brainreg output folder found.

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv[1:].

    Returns:
        int: 0 on success, 1 when no folders were found or any copy failed.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    overwrite = "--overwrite" in argv
    paths = [arg for arg in argv if not arg.startswith("--")]
    if not paths:
        print(__doc__)
        return 1

    folders = list(collect_output_folders(paths))
    if not folders:
        print(
            "No brainreg output folders found (folders containing "
            f"{TIFF_BASENAMES[0]} and {TIFF_BASENAMES[1]})."
        )
        return 1

    written = skipped = failed = 0
    for folder in folders:
        for base in TIFF_BASENAMES:
            src = os.path.join(folder, base)
            if not os.path.isfile(src):
                continue
            try:
                dst = write_ng_copy(src, overwrite=overwrite)
            except Exception as e:
                failed += 1
                print(f"FAILED {src}: {e}")
                continue
            if dst is None:
                skipped += 1
                print(f"SKIPPED (exists) {src}")
            else:
                written += 1
                print(f"WROTE {dst}")
    print(f"Done: {written} written, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
