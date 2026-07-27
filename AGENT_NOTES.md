# Agent Notes

- 2026-06-12: Updated backend `AGENTS.md` to require reading root and backend notes before changes and appending backend notes after backend file changes.
- 2026-06-15: Added backend reader plugin `reader_plugins/ome_zarr_reader/` with metadata extraction, TIFF slice export, and 100um volume generation for `.ome.zarr` datasets stored as top-level `scale0`, `scale1`, etc. arrays.
- 2026-07-23: Updated `start_pipeline.py` so per-job parse/validation/runtime failures return `err` status, failed moves are logged instead of raised, and unexpected exceptions no longer break the long-running watcher loop.
