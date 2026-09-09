# Agent Notes

- 2026-06-12: Updated backend `AGENTS.md` to require reading root and backend notes before changes and appending backend notes after backend file changes.
- 2026-06-15: Added backend reader plugin `reader_plugins/ome_zarr_reader/` with metadata extraction, TIFF slice export, and 100um volume generation for `.ome.zarr` datasets stored as top-level `scale0`, `scale1`, etc. arrays.
- 2026-08-27: Added optional backend job/workflow history updates behind `PEACE_ENABLE_JOB_HISTORY`. The watcher now records dispatching failures, persists returned SLURM job IDs/provenance/log folders for single jobs and readers, and stores per-step workflow dispatch metadata needed for Home-page tracking and forked reruns from a chosen step.
- 2026-09-02: Added validated preprocessing z-range utilities and absolute-index SLURM submission across stretch, histogram equalization, blur, gamma, resize, image calculator, Cellpose denoise, FFT stripe removal, and rembg; subset outputs retain source z-indices and publish `processed_z_range` provenance.
- 2026-09-02: Reused unsuffixed Cellpose raw min/max caches for full-availability inputs and reserved range suffixes for genuinely sparse upstream inputs.
