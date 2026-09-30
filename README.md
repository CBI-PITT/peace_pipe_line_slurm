# PEACE back-end pipeline (SLURM)

A long-running daemon that watches JSON task folders on a shared filesystem and
submits whole-organ image-analysis jobs to a SLURM cluster. It is the execution
half of the PEACE system: a Flask web app (`../generate_peace_json_test/`)
writes task files, this daemon validates and executes them. All operations work
on folders of TIFF slices — the common medium that every reader plugin converts
data to, and that every analysis operation consumes.

## The PEACE system

```
                       researcher (web browser)
                              │
                              ▼
              ┌───────────────────────────────┐
              │        Flask web app          │
              │  (generate_peace_json_test)   │
              └───────┬───────────────┬───────┘
              writes  │               │ mounts
                      ▼               ▼
         ┌──────────────────┐  ┌────────────────────────┐
         │ JSON task files  │  │     file browser       │
         │ (shared folder)  │  │ (flask_file_browser_   │
         └────────┬─────────┘  │ test) + Add-to-        │
                  │ polls      │ dashboard button       │
                  ▼            └───────────┬────────────┘
    ┌──────────────────────┐              │
    │  back-end daemon     │              ▼
    │  (peace_pipe_line_   │   ┌───────────────────────┐
    │  slurm_test)         │   │    data dashboard     │
    └──────────┬───────────┘   │    (data_dashboard)   │
               │ sbatch        └───────────▲───────────┘
               ▼                           │ results
    ┌──────────────────────┐               │
    │    SLURM cluster     │──── outputs ──┘
    │  (job arrays, GPUs)  │
    └──────────────────────┘
```

| Repository | Role |
|---|---|
| `../generate_peace_json_test/` | Flask web app that generates task JSON |
| `../flask_file_browser_test/` | File-browser blueprint used by the web app |
| `../data_dashboard/` | Dashboard that visualizes pipeline results |
| **`peace_pipe_line_slurm_test` (this repo)** | Back-end daemon that runs the tasks |

## How it works

1. **Poll** — the daemon scans its watched folders every few seconds, in a
   fixed order: reader tasks (`SLURM_reader*.json`), single-operation tasks
   (`SLURM_settings*.json`), then workflows (`SLURM_workflow*.json`).
2. **Validate** — before any cluster resource is touched: the JSON must parse
   and carry an `extras` dictionary; the `user` field must match a strict
   account pattern; all input/output paths must resolve inside that user's
   sandbox on the filesystem (`<FS_ROOT>/<user>/`, real-path resolution with
   symlink-escape rejection); the requested operation must resolve against the
   whitelisted registry.
3. **Bootstrap provenance** — TIFF-series inputs are guaranteed to carry a
   `.dataset_info.json` file; one is created from the image data if absent.
4. **Dispatch** — the operation is instantiated with
   `operation(input, output, **extras)`; its `run()` writes SLURM job scripts,
   submits them, and returns `(provenance_path, job_ids)`.
5. **Archive** — the task file is renamed into `done/` (or `err/` with the
   exception logged). Failures are isolated: one bad submission never stops
   the daemon.

Workflows are expanded by the daemon itself: steps are dispatched in order,
each step's `input_bindings` are resolved by loading the provenance of earlier
steps, and the SLURM job IDs of all steps accumulate into `--depend=afterok`
chains so the scheduler enforces ordering.

## JSON task contract

Single operation (`SLURM_settings*.json`) or reader (`SLURM_reader*.json`):

```json
{
    "input": "/data/iyer-s/RSCM/brain1_mag8x_montage.ims",
    "output": "/data/iyer-s/analysis/brain1/tiffs/",
    "operation": "spotiflow",
    "extras": {
        "user": "iyer-s",
        "priority": "2",
        "signal_channel": 1,
        "resolution_level": 1,
        "model": "general",
        "with_dbscan": false,
        "prerequisites": [123456]
    }
}
```

Workflow (`SLURM_workflow*.json`) — `input` becomes a bindings dict and
`output` becomes an output name that later steps can bind to:

```json
{
    "workflow_id": "…",
    "steps": [
        {"input_bindings": {}, "output_name": "tiffs", "operation": "ome_zarr_reader",
         "extras": {"user": "iyer-s", "channel": 0, "resolution_level": 4}, "step_id": "…"},
        {"input_bindings": {"input": "tiffs"}, "output_name": "reg", "operation": "brainreg",
         "extras": {"user": "iyer-s", "atlas": "allen_mouse_25um", "brain_geometry": "full"}, "step_id": "…"}
    ]
}
```

Top-level keys are a compatibility boundary with the web app: `input`,
`output`, `operation`, `extras` (a dictionary, unpacked as keyword arguments).
CSV-input operations (`dbscan`, `transform_points`, `nearest_neighbor`,
`mean_intensity`, `resnet_classification`, `combine_with_metadata`) may submit
empty `input`/`output` and derive paths from the cells-CSV provenance via
`extras.cells_path` / `extras.cell_candidates_path`.

## Operations

24 built-in operations in six functional categories, plus the reader plugins.
All operations consume and produce the shared representations: TIFF-series
folders (`r{rl}_t00_c{ch}_z{zzzz}.tif` per-z-plane names) and CSV point tables.

| Category | Operations | Input → Output |
|---|---|---|
| Pre-processing | `stretch_contrast`, `adaptive_histogram_equalization`, `gaussian_blur`, `gamma_correction`, `resize_image`, `image_calculator`, `denoise_cellpose`, `remove_stripes_fft`, `remove_background` | TIFF → TIFF (+ masks) |
| Cell detection | `deepblink`, `spotiflow`, `cellfinder` | TIFF → point CSVs |
| Segmentation | `cellpose`, `ilastik`, `unet_3d` | TIFF → mask TIFF series |
| Registration | `brainreg`, `ants` | TIFF → atlas-aligned volumes + deformation fields |

`brainreg` additionally writes BrAinPI-friendly copies
(`downsampled_ng.tif`, `boundaries_ng.tif`) into the registration folder after
each run — OME-TIFFs with declared `ZYX` axes (the plain originals report
`QYX`, which BrAinPI's loader rejects) so the PEACE File Browser's
"Neuroglancer" button can serve them. The zip step runs before the copies, so
`registration_<atlas>.zip` stays napari-oriented. The same script
(`operations/brainreg/write_ng_copies.py`) works standalone for existing
outputs: `python write_ng_copies.py <brainreg output folder or any parent>
[--overwrite]`.
| Post-processing | `dbscan`, `resnet_classification`, `delete_background_detections`, `transform_points`, `mean_intensity`, `nearest_neighbor` | point CSVs, masks → annotated CSVs |
| Metadata | `combine_with_metadata` | point CSVs → CSVs with experiment covariates |

Reader plugins (format conversion):

| Reader | Converts |
|---|---|
| `imaris_reader` / `imaris_reader_crop` | Imaris `.ims` (multi-resolution pyramids, cropped sub-volumes) |
| `ome_zarr_reader` | OME-Zarr stores |
| `omehans_reader` | OMEhans HDF5 volumes |
| `jp2_reader` | JPEG-2000 slices |
| `tiff_series_reader` | pass-through + provenance generation |

## Plugin architecture

Two abstract base classes in `operations/base/`: `ImageOperation` and
`ImageReader`, each defining a constructor that receives the input path, output
path and `extras` keyword arguments, and an abstract `run()` that returns
`(provenance_path, job_ids)`. Built-in operations live in `operations/<name>/`
(pairs an orchestration `main.py` with the `do_*.py` workers the generated jobs
run). Additional operations and readers are plugins: self-contained
subdirectories under `plugins/` and `reader_plugins/` that are discovered
dynamically at start-up and whitelisted before dispatch — extending PEACE
requires no changes to the daemon.

## Provenance

Every input and output folder carries a `.dataset_info.json` file recording the
source dataset, resolution, shape, channels, orientation, all operation
parameters, and the input/output paths. Each operation extends a `sequence`
field with its own name (e.g. `imaris_reader,contrast_stretched,spotiflow`),
and downstream operations resolve volume-level facts (full-resolution shape,
raw source path, voxel size) by walking the `source` chain back to the original
acquisition. Provenance drives workflow chaining, the web app's output-path
autofill, and FAIR-style reproducibility.

## SLURM integration

- **Single jobs** (`sbatch`), **job arrays** (`--array=0-N`) across z-planes or
  volume chunks (e.g. 40×1700×3500-voxel chunks for detector inference), and
  **partial arrays** that resubmit only the indices whose outputs are missing —
  an effective resume mechanism after preemption or node failure.
- **Dependencies** — workflow steps submit with `--depend=afterok:...` and
  `--kill-on-invalid-dep=yes`.
- **Priorities** — a per-user priority (0–5) maps to nice values scaled by
  core count, so administrators can balance politeness against throughput.
- **Partitions** — CPU, GPU and high-memory queues; GPU generic resources are
  requested only on GPU-enabled partitions.
- **Log triage** — job logs are scanned for common failure signatures
  (tracebacks, segmentation faults, missing modules, permission errors) and
  failing logs are moved into an `errors/` folder.
- **Safety** — every user-derived value is shell-quoted (`shlex.quote`) when
  interpolated into generated job scripts; runtime variables such as
  `$SLURM_ARRAY_TASK_ID` are left untouched.

## Install and run

```bash
python3 -m pip install -r requirements.txt
python3 start_pipeline.py
```

`requirements.txt` pins the analysis packages the daemon itself needs
(`cellfinder`, `opencv-python`, `imaris_ims_file_reader`, `brainrender`,
`dask`). Individual operations run in their own conda environments, activated
inside the generated job scripts — see below.

### Configuration (`analysis/settings.py`)

| Key | Meaning |
|---|---|
| `JSON_FOLDERS` | list of watched task folders (e.g. `./json`) |
| `FS_ROOT` | filesystem root of the per-user processing sandbox |
| `SLURM_PARTITION_CPU` / `_GPU` / `_HIGH_RAM` / `_EXTREME` | partition names |
| `SLURM_JOBS_NICE_LEVEL`, `PRIORITY_TO_NICE_MAP_*` | scheduling politeness |
| `UMASK` | permissions for created folders (group-writable collaboration) |
| `INFO_FILE_NAME` | provenance file name (`.dataset_info.json`) |
| `ENABLE_JOB_HISTORY`, `JOB_HISTORY_DIR` | optional history records consumed by the web app |

### Conda environments

Operations run in isolated environments, activated inside each generated job
script. Adapt the names/paths to your site (they are configured where job
scripts are generated, per operation):

| Operation | Environment | Key dependencies |
|---|---|---|
| deepblink / spotiflow | `deepblink` | python 3.8, TensorFlow 2.8 GPU (cudatoolkit 11.0, cudnn 8.2), deepblink, chardet |
| cellfinder / resnet_classification | `cellfinder` | python 3.9, cellfinder 0.4.21, bg_space, brainreg 0.4, importlib_metadata<8.0 |
| brainreg / transform_points | `brainreg` | python 3.9, brainreg 0.4, numpy 1.22 |
| ants | `ants` | python 3.9, antspyx, bg_atlasapi, bg_space, scikit-image |
| cellpose / denoise_cellpose | `cellpose` | cellpose (GPU optional) |
| remove_background | `remove_bg_py311_cuda124` | python 3.11, PyTorch + CUDA 12.4, SAM 2 |
| imaris/ome-zarr/jp2 readers | `peace` / `omehans-reader` | imaris_ims_file_reader, zarr, pillow |
| everything else (daemon, pre-processing, post-processing) | `peace` | this repo's `requirements.txt` |

## Adding an operation

1. Create `operations/<name>/` with a `main.py` orchestrator (subclass
   `ImageOperation`; prepare folders, provenance, SLURM scripts, submission)
   and a `do_*.py` worker (the per-task computation the job runs).
2. Register the class in `operations/__init__.py` — the name must match the
   `operation` string in the task JSON.
3. Record all parameters in the provenance and extend the `sequence` field.
4. Keep the worker's CLI argument order in sync with what `main.py` emits.
5. Verify: `python3 -m py_compile start_pipeline.py operations/*/main.py operations/*/do_*.py`,
   then add a matching form/plugin in the web app
   (`../generate_peace_json_test/`) so the operation is usable from the UI.

## Tests

```bash
python3 -m pytest
```

Seven test modules (~60 tests) cover validation (user regex, realpath
containment, operation whitelist), SLURM submission (array ranges, resume
skips, failure paths), `shell_arg` quoting (including a real bash behavioral
test), z-range handling, job history, and log triage. The suite is import-light
and uses mocked `subprocess` calls and temporary directories, so it runs
without SLURM or the analysis environments.

## Security model

Tasks are executed on behalf of web users, so dispatch is hardened: validated
usernames, per-user sandboxes with real-path containment (symlink escapes
rejected), a whitelisted operation registry, and shell-quoted job scripts.
Files are created group-writable (`umask 0o006`) for collaboration. Run the
daemon as an unprivileged service account with access to the sandbox root and
the SLURM scheduler.
