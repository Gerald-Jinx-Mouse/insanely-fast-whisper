# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `src/insanely_fast_whisper/`. The CLI entrypoint is `src/insanely_fast_whisper/cli.py`, and diarization/result helpers live under `src/insanely_fast_whisper/utils/`. Use `convert_output.py` to turn JSON transcripts into `srt`, `vtt`, or `txt`. Benchmark and demo notebooks are in `notebooks/`. The `tests/` package exists but is currently only a placeholder, so new feature work should add focused tests there.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode and registers the `insanely-fast-whisper` CLI locally.
- `insanely-fast-whisper --help` is the fastest smoke test for argument parsing and entrypoint wiring.
- `python convert_output.py output.json -f srt -o .` validates the transcript conversion helper against a real JSON output file.
- `python -m build` creates source and wheel artifacts from `pyproject.toml` when preparing a release.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions/variables/modules, and short, explicit argument names that mirror CLI flags. Keep imports grouped at the top and prefer small helper functions in `utils/` over adding more logic to `cli.py`. The repository does not currently define `ruff`, `black`, or `mypy` settings, so preserve the surrounding style when editing and avoid broad formatting-only diffs.

## Testing Guidelines
No automated suite or coverage gate is configured yet. Add tests under `tests/` using `test_<feature>.py` naming, and prefer `pytest`-style assertions for new coverage. For GPU-dependent changes, include a manual smoke-test command in the PR description, such as `insanely-fast-whisper --file-name sample.wav --device-id 0`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, sometimes with prefixes like `feat:`, `doc:`, or `docs:`. Keep commits narrowly scoped, for example `feat: validate speaker count arguments`. PRs should explain the user-visible change, note any model/device assumptions, link related issues, and include example CLI output when behavior or transcript formatting changes.

## Security & Configuration Tips
Do not hardcode Hugging Face tokens or local file paths. Keep large audio samples, generated transcripts, and notebook outputs out of the repository unless they are essential fixtures for a new test.
