# Agent Notes

Use this file as persistent working memory for agent changes in this repository.
Read it before making code changes.
Append new notes; do not delete older entries unless explicitly asked.

## 2026-06-05
- Task: Add persistent agent notes workflow for this repository.
- Files changed: `AGENTS.md`, `AGENT_NOTES.md`
- Important decisions: Use a single repo-root notes file named `AGENT_NOTES.md`; require agents to read it before changes and append to it afterward.
- Follow-up items: Keep this file updated on future code changes; if note format needs to evolve, update `AGENTS.md` and preserve older entries.

## 2026-06-05
- Task: Replace per-operation conda activation with container execution and add configurable apptainer/singularity runtime support.
- Files changed: `analysis/settings.py`, `utils/containers.py`, `operations/resize_image/main.py`, `operations/stretch_contrast/main.py`, `operations/gaussian_blur/main.py`, `operations/gamma_correction/main.py`, `operations/adaptive_histogram_equalization/main.py`, `operations/image_calculator/main.py`, `operations/combine_with_metadata/main.py`, `operations/delete_background_detections/main.py`, `operations/remove_stripes_fft/main.py`, `operations/deepblink/main.py`, `operations/spotiflow/main.py`, `operations/denoise_cellpose/main.py`, `operations/unet_3d/main.py`, `operations/resnet_classification/main.py`, `operations/cellpose/main.py`, `operations/transform_points/main.py`, `operations/dbscan/main.py`, `operations/ilastik/main.py`, `operations/ants/main.py`, `operations/delete_background_detections/process_all_z_layers.py`, `operations/remove_stripes_fft/run_all_steps.py`, `operations/deepblink/run_all_steps.py`, `operations/spotiflow/run_all_steps.py`, `operations/denoise_cellpose/run_all_steps.py`, `operations/resnet_classification/do_classification.py`, `reader_plugins/imaris_reader/main.py`, `reader_plugins/imaris_reader_crop/main.py`, `reader_plugins/omehans_reader/main.py`, `plugins/rembg/main.py`, `operations/cellfinder/main.py`, `operations/brainreg/main.py`, `operations/deepblink/process_one_chunk.py`, `operations/cellpose/process_one_chunk.py`, `operations/unet_3d/process_one_chunk.py`, `reader_plugins/jp2_reader/main.py`.
- Important decisions: Added global `CONTAINER_RUNTIME` with allowed values `apptainer` and `singularity`; added shared container helpers with optional local `containers.json` support and fallback to `<env_name>.sif`; GPU launches now add `--nv` only where jobs explicitly request GPU execution.
- Follow-up items: Contributor docs should eventually describe the optional `containers.json` format; runtime validation currently happens when launch commands are built, so missing `.sif` files fail at task-generation time rather than later in SLURM.

## 2026-06-05
- Task: Add a shared fallback container directory for large `.sif` files.
- Files changed: `analysis/settings.py`, `utils/containers.py`
- Important decisions: Added `CONTAINER_FALLBACK_DIR = None` to settings; container resolution order is now local `containers.json` path if present, local `<env_name>.sif`, then `<CONTAINER_FALLBACK_DIR>/<env_name>.sif` when configured.
- Follow-up items: Set `CONTAINER_FALLBACK_DIR` on deployed clusters; missing-container errors now list all checked paths to simplify debugging.
