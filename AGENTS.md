# AGENTS Guide

## Scope
This repository is the backend half of the PEACE pipeline.
It watches JSON task files and submits image-processing work to SLURM.
There is a closely related sibling app at `/h20/CBI/Iana/src/peace/generate_peace_json_test`.
That Flask app generates the JSON consumed here.
Treat the JSON schema between the two apps as a compatibility boundary.

## Related Rule Files
Checked in this workspace on 2026-06-05:

- Local `AGENTS.md` exists in this repo and in the sibling Flask repo.
- No `.cursorrules` files found.
- No `.cursor/rules/` directory found.
- No `.github/copilot-instructions.md` file found.

If any of those files are added later, update this guide.

## Agent Notes
Before making code changes, read `AGENT_NOTES.md` in the repository root.
After completing changes, append a short note with:

- date
- task summary
- files changed
- important decisions
- follow-up items

Treat `AGENT_NOTES.md` as persistent working memory for future agents.
Do not rewrite or remove prior notes unless explicitly asked.

## Repository Layout
- `start_pipeline.py`: main watcher and JSON entry point.
- `operations/`: built-in backend operations loaded by import.
- `plugins/`: extra backend operations loaded dynamically.
- `reader_plugins/`: dynamic reader plugins.
- `operations/__init__.py`: registration list for built-in operations.
- `operations/base/__init__.py`: `ImageOperation` and `ImageReader` base classes.
- `analysis/settings.py`: runtime paths, SLURM settings, and constants.
- `utils/slurm.py`: SLURM submission helpers and log parsing.
- `README.md`: environment-specific usage notes.

## Runtime Model
The backend expects task JSON with these top-level keys:

```json
{
  "input": "/path/to/input",
  "output": "/path/to/output",
  "operation": "stretch_contrast",
  "extras": {}
}
```

Important contract rules:
- `extras` must be a dictionary.
- Operation names must match import or plugin names exactly.
- New form fields in the Flask app must match backend `kwargs.get(...)` names.
- Do not rename JSON fields casually.

## Install And Run
Install backend dependencies from this repository root:

```bash
python3 -m pip install -r requirements.txt
```

Start the watcher locally:

```bash
python3 start_pipeline.py
```

- Many operations assume SLURM, shared filesystem paths, and external conda envs exist.
- Local development often supports syntax checks and code inspection better than full execution.

## Build, Lint, And Test Commands
There is no formal build system in this repo.
There is also no configured linter, formatter, `pytest`, `tox`, `pyproject.toml`, `pytest.ini`, or `setup.cfg`.
No test files were found in this repository during inspection.

Use `py_compile` as the required lightweight validation step.
Syntax check one file:
```bash
python3 -m py_compile path/to/file.py
```
Syntax check several touched files:
```bash
python3 -m py_compile start_pipeline.py utils/slurm.py operations/stretch_contrast/main.py
```
Syntax check all backend operation entrypoints:
```bash
python3 -m py_compile start_pipeline.py operations/*/main.py operations/*/do_*.py reader_plugins/*/*.py plugins/*/*.py
```
Closest equivalent to a single test today:
```bash
python3 -m py_compile path/to/touched_file.py
```
If a real test suite is added later, use normal single-test commands such as:
```bash
pytest path/to/test_file.py::test_name
```

## Manual Validation
For backend changes:
- Create or inspect a representative JSON task.
- Verify `operation` matches a built-in import or plugin class name.
- Verify `extras` keys line up with the operation constructor and runtime lookups.
- Verify `main.py` emits CLI arguments in the same order worker scripts expect.
- Verify output provenance still writes `.${INFO_FILE_NAME}` metadata correctly.

For changes that affect the Flask generator indirectly:
- Confirm the backend still accepts JSON produced by the sibling Flask app.
- Spot-check any new parameter names on both sides of the contract.

## Change Expectations
- Keep changes minimal and local.
- Preserve compatibility with the sibling Flask app.
- Follow the existing operation pattern instead of introducing new abstractions.
- When adding a new operation, update registration, provenance, SLURM script generation, and worker CLI together.
- Avoid broad refactors unless they are required to complete the task safely.

Typical backend operation shape:
1. `main.py` prepares folders, provenance, and SLURM submission.
2. `do_*.py` performs per-image or per-task processing.
3. `__init__.py` exposes the operation class.
4. `operations/__init__.py` registers built-in operations.

## Code Style
Follow the style already present in each file.
This codebase is mostly imperative Python with explicit state and path handling.
### Imports
- Prefer standard library, then third-party, then local imports.
- Keep imports simple and readable; one per line is preferred when practical.
- Use relative imports inside operation packages when neighboring files already do.
- Do not add unused imports.

### Formatting
- Use 4-space indentation.
- Keep blank lines and wrapping similar to surrounding code.
- Prefer explicit multi-line dicts and function calls over dense one-liners.
- Keep string quoting consistent with the file you are editing.
- Default to ASCII unless the file already requires non-ASCII.

### Naming
- Respect existing naming even when inconsistent.
- Backend operation class names are lower-case and match operation identifiers, for example `stretch_contrast`.
- Base classes use CapWords, for example `ImageOperation` and `ImageReader`.
- Directory names, operation names, and registration imports must stay aligned.
- JSON keys should remain lower-case with underscores.

### Types
- There is no strong type-hinting convention here.
- Do not add broad annotations just to modernize code.
- Prefer straightforward control flow and clear variable names.

### Error Handling
- Fail early on invalid inputs that would create broken SLURM jobs.
- Raise specific built-in exceptions when adding new validation.
- Preserve existing logging behavior in long-running entry points.
- Do not silently swallow errors unless the surrounding module already does so intentionally.
- Follow the existing JSON parsing and plugin-loading patterns in `start_pipeline.py`.

### Files, Paths, And Shell Commands
- Prefer `os.path`; it is the dominant convention here.
- Preserve existing `umask`, directory creation, and provenance-writing behavior.
- Quote paths with spaces when generating shell commands.
- Be careful with SLURM command construction and dependency flags.

### Dependencies And External Tools
- Do not assume SLURM or cluster-only tools are available locally.
- Do not replace environment-specific commands unless the task requires it.
- Keep external package assumptions consistent with `requirements.txt` and README notes.

## Agent Reminders
- Read the touched operation end-to-end before editing it.
- Read `AGENT_NOTES.md` before editing and append to it after each change set.
- Check sibling worker scripts whenever `main.py` CLI arguments change.
- Protect the JSON contract first; convenience refactors are secondary.
- Run `python3 -m py_compile` on every touched Python file before finishing.
