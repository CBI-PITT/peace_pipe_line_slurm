# AGENTS Guide

## Scope

This workspace contains two related Python projects:

- `peace_pipe_line_slurm_test`: backend pipeline that watches JSON folders and submits SLURM jobs.
- `generate_peace_json_test`: Flask app that generates those JSON task files.

Treat them as two separate apps that must stay compatible at the JSON contract level.

## Existing Agent Instructions

There is currently no existing `AGENTS.md` in this workspace.

There are also no repo-local agent rule files present:

- No `.cursorrules`
- No `.cursor/rules/`
- No `.github/copilot-instructions.md`

If those files are added later, update this document to reflect them.

## Repository Layout

- Root: `/h20/CBI/Iana/src/peace`
- Backend pipeline: `/h20/CBI/Iana/src/peace/peace_pipe_line_slurm_test`
- Flask JSON generator: `/h20/CBI/Iana/src/peace/generate_peace_json_test`

Important backend locations:

- `start_pipeline.py`: long-running JSON watcher and entry point
- `operations/`: built-in pipeline operations
- `plugins/`: backend plugin operations
- `reader_plugins/`: backend reader plugins
- `analysis/settings.py`: runtime paths and pipeline settings
- `utils/slurm.py`: SLURM submission helpers

Important Flask locations:

- `flask_app/app.py`: Flask entry point
- `flask_app/forms.py`: WTForms definitions
- `flask_app/operations/`: UI operation plugins
- `flask_app/templates/`: form and page templates
- `flask_app/workflows/`: workflow JSON generation

## Install Commands

Backend pipeline:

```bash
python3 -m pip install -r requirements.txt
```

Flask app:

```bash
python3 -m pip install -r requirements.txt
```

Notes:

- The READMEs also reference environment-specific conda setups on the target infrastructure.
- Many backend operations assume external tools, SLURM, and specific conda envs exist.

## Run Commands

Run the backend watcher locally from `peace_pipe_line_slurm_test`:

```bash
python3 start_pipeline.py
```

Run the Flask app locally from `generate_peace_json_test/flask_app`:

```bash
python3 app.py
```

The Flask app binds to `0.0.0.0:1212` in the current code.

## Build / Lint / Test Commands

There is no formal build system in this workspace.

There is also no configured linter, formatter, `pytest`, `tox`, `pyproject.toml`, `pytest.ini`, or `setup.cfg` here.

Use these validation commands instead.

### Syntax Check

Check a single Python file:

```bash
python3 -m py_compile path/to/file.py
```

Check multiple touched files:

```bash
python3 -m py_compile file1.py file2.py file3.py
```

Recommended for backend changes:

```bash
python3 -m py_compile start_pipeline.py operations/**/main.py operations/**/do_*.py
```

Recommended for Flask changes:

```bash
python3 -m py_compile flask_app/app.py flask_app/forms.py flask_app/operations/*.py
```

### Tests

There is currently no automated test suite checked into this workspace.

Because there are no tests, there is no real “single test” command to run today.

If a test file is added later and `pytest` is introduced, use the normal single-test form:

```bash
pytest path/to/test_file.py::test_name
```

Until then, treat targeted `py_compile` plus manual validation as the required verification path.

### Manual Validation

For Flask UI changes:

- Start `flask_app/app.py`
- Load the affected form in the browser
- Submit a task
- Inspect the generated JSON in the configured JSON folder

For backend pipeline changes:

- Create or inspect a representative JSON task
- Verify the backend operation name matches an import in `operations/__init__.py`
- Verify the operation class accepts the generated `extras`
- Verify worker scripts accept the CLI arguments emitted by `main.py`

## Change Workflow Expectations

- Keep changes minimal and local.
- Preserve compatibility between Flask JSON generation and backend JSON consumption.
- When adding a new pipeline operation, usually update both apps.
- For new preprocessing operations, mirror the existing pattern used by `stretch_contrast` and similar operations.

Typical full-stack operation addition:

1. Add a form class in `flask_app/forms.py`
2. Add a Flask operation plugin in `flask_app/operations/`
3. Choose the correct template, often `form_autofill_output.html` or a custom one
4. Add a backend operation package in `peace_pipe_line_slurm_test/operations/`
5. Register it in `peace_pipe_line_slurm_test/operations/__init__.py`
6. Ensure provenance, SLURM script generation, and worker CLI stay aligned
7. Run `python3 -m py_compile` on all touched Python files

## Code Style

Follow the existing codebase style rather than imposing a new framework-wide style.

### Python Version and General Style

- Use Python 3 syntax.
- Prefer simple module-level functions and classes.
- Keep logic explicit and imperative.
- Avoid introducing heavy abstractions unless there is repeated need.
- Match the current file’s style when editing older modules.

### Imports

- Group imports in this order when practical:
  1. standard library
  2. third-party packages
  3. local project imports
- Prefer one import per line for readability.
- Use relative imports inside backend operation packages when the surrounding code already does so, for example `from ..base import ImageOperation`.
- Do not add unused imports.

### Formatting

- Use 4-space indentation.
- Keep blank lines similar to nearby code.
- Prefer readable multi-line dicts and function calls over dense one-liners.
- Keep string quoting consistent with the surrounding file.
- Avoid introducing non-ASCII characters unless already required.

### Naming

- Respect existing naming conventions even when inconsistent.
- Backend operation class names are currently lower-case and match operation identifiers, for example `class stretch_contrast(ImageOperation)`.
- Flask operation plugin classes use CapWords, for example `class StretchContrast(BaseOperation)`.
- Form classes use CapWords and end with `Form`.
- Keep JSON field names stable and lower-case with underscores.
- Match operation folder names, operation string names, and registration imports exactly.

### Types

- There is no established type-hinting convention in this codebase.
- Do not add extensive type annotations unless you are already editing a typed area.
- Prefer clear variable names and straightforward control flow over introducing partial typing.

### Error Handling

- Fail early for invalid inputs that would otherwise create broken SLURM jobs.
- Raise specific built-in exceptions when possible, such as `FileNotFoundError` or `ValueError`.
- In long-running entry points, preserve the existing logging and exception behavior.
- Do not silently swallow errors in new code unless the surrounding module already does that intentionally.
- For JSON parsing and plugin lookup, follow the existing backend pattern in `start_pipeline.py`.

### File and Path Handling

- Use `os.path` consistently; that is the dominant pattern here.
- Quote paths with spaces when generating shell commands.
- Keep file permissions behavior intact where the existing code sets `umask` or `chmod`.
- Preserve the backend convention of writing `.dataset_info.json` provenance files in output folders.

### JSON Contract Rules

- Flask emits JSON with top-level `input`, `output`, `operation`, and `extras`.
- Backend expects `extras` to be a dictionary.
- Do not rename JSON keys casually; treat them as a compatibility boundary.
- When adding new form fields, make sure backend `kwargs.get(...)` names match the form field names.

### SLURM Operation Pattern

When adding or editing backend operations:

- `main.py` should prepare folders, provenance, and SLURM submission.
- `do_*.py` should do the per-task image processing work.
- Keep worker CLI arguments in the same order as `main.py` emits them.
- Support partial resume if the neighboring operation pattern already does.
- Record operation parameters in provenance.

### Flask UI Pattern

- Prefer using existing form templates unless the UX requires a custom layout.
- Use `form_autofill_output.html` when output should be derived from the first input path.
- If a field must support both browsing and manual entry, a custom template is acceptable.
- Keep operation categories aligned with how `app.py` groups operations.

## What Not To Do

- Do not invent a lint or formatter workflow that the repo does not use.
- Do not add broad refactors while implementing a single operation.
- Do not break the JSON schema between the Flask app and backend.
- Do not assume SLURM, external data paths, or conda envs are available in local development.

## Preferred Verification Summary

For most changes in this repo, the expected verification is:

1. `python3 -m py_compile` on every touched Python file
2. manual inspection of generated JSON when changing Flask forms/plugins
3. manual review of CLI argument alignment between backend `main.py` and worker scripts
4. manual spot-check of provenance fields for new operation parameters
