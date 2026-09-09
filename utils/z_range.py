import os
import re
from glob import glob


Z_FILENAME_PATTERN = re.compile(r"_z(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
Z_INDEX_PATTERN = re.compile(r"_z(\d+)\.[^.]+$", re.IGNORECASE)


def get_available_z_range(metadata):
    total_z = int(metadata["shape"][-3])
    processed = metadata.get("processed_z_range")
    if not processed:
        return 0, total_z

    return int(processed["start"]), int(processed["end"])


def normalize_z_range(metadata, z_start=0, z_end=-1):
    if isinstance(z_start, bool) or isinstance(z_end, bool):
        raise ValueError("z_start and z_end must be integers")
    if isinstance(z_start, float) and not z_start.is_integer():
        raise ValueError("z_start and z_end must be integers")
    if isinstance(z_end, float) and not z_end.is_integer():
        raise ValueError("z_start and z_end must be integers")

    try:
        requested_start = int(z_start)
        requested_end = int(z_end)
    except (TypeError, ValueError):
        raise ValueError("z_start and z_end must be integers")

    total_z = int(metadata["shape"][-3])
    resolved_end = total_z if requested_end == -1 else requested_end

    if requested_start < 0:
        raise ValueError("z_start must be at least 0")
    if requested_end < -1:
        raise ValueError("z_end must be -1 or a non-negative integer")
    if requested_start >= resolved_end:
        raise ValueError("z_start must be less than z_end")
    if resolved_end > total_z:
        raise ValueError(
            f"z_end {resolved_end} exceeds the source depth {total_z}"
        )

    available_start, available_end = get_available_z_range(metadata)
    if requested_start < available_start or resolved_end > available_end:
        raise ValueError(
            f"Requested z range [{requested_start}, {resolved_end}) is outside "
            f"the available range [{available_start}, {available_end})"
        )

    return {
        "requested_start": requested_start,
        "requested_end": requested_end,
        "start": requested_start,
        "end": resolved_end,
        "total_z": total_z,
        "is_full": requested_start == 0 and resolved_end == total_z,
    }


def z_range_suffix(selection):
    if selection["is_full"]:
        return ""
    return f"_z{selection['start']}-{selection['end']}"


def z_range_provenance(selection):
    return {
        "start": selection["start"],
        "end": selection["end"],
        "end_exclusive": True,
    }


def existing_z_indices(folder, extension="tif"):
    indices = set()
    if not os.path.isdir(folder):
        return indices

    extension = extension.lower().lstrip(".")
    for filename in os.listdir(folder):
        if extension and not filename.lower().endswith(f".{extension}"):
            continue
        match = Z_INDEX_PATTERN.search(filename)
        if match:
            indices.add(int(match.group(1)))

    return indices


def resolve_tiff_path(folder, metadata, resolution_level, channel, z):
    canonical_path = os.path.join(
        folder,
        f"r{int(resolution_level):02d}_t00_c{int(channel):02d}_z{int(z):04d}.tif"
    )
    if os.path.exists(canonical_path):
        return canonical_path

    files = sorted(
        glob(os.path.join(folder, "*.tif"))
        + glob(os.path.join(folder, "*.tiff"))
    )
    indexed = {}
    unindexed = []
    for path in files:
        match = Z_FILENAME_PATTERN.search(os.path.basename(path))
        if match:
            indexed[int(match.group(1))] = path
        else:
            unindexed.append(path)

    if indexed:
        if unindexed:
            raise ValueError(f"Cannot mix indexed and unindexed TIFF filenames in {folder}")
        if z not in indexed:
            raise FileNotFoundError(f"No TIFF file found for z index {z} in {folder}")
        return indexed[z]

    if metadata is None:
        raise FileNotFoundError(f"No TIFF file found for z index {z} in {folder}")

    available_start, available_end = get_available_z_range(metadata)
    if len(files) != available_end - available_start:
        raise ValueError(
            f"Cannot assign absolute z indices to {len(files)} TIFF files in {folder}"
        )
    position = z - available_start
    if position < 0 or position >= len(files):
        raise FileNotFoundError(f"No TIFF file found for z index {z} in {folder}")
    return files[position]
