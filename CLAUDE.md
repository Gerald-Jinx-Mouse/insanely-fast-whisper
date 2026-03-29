# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
python -m pip install -e .          # Install in editable mode, registers `insanely-fast-whisper` CLI
insanely-fast-whisper --help        # Smoke test for argument parsing and entrypoint wiring
python convert_output.py output.json -f srt -o .  # Convert JSON transcript to SRT/VTT/TXT
python -m build                     # Create source and wheel artifacts for release
```

No automated test suite exists yet. Tests go in `tests/` using `test_<feature>.py` naming with pytest-style assertions. For GPU-dependent changes, include a manual smoke-test command in the PR description.

## Architecture

This is a CLI tool wrapping Hugging Face Transformers' speech recognition pipeline with speaker diarization support.

**Flow:** CLI arg parsing → `transformers.pipeline("automatic-speech-recognition")` → optional pyannote diarization → JSON output

**Module graph:**
```
cli.py                          # Entry point, arg parsing, pipeline setup, output writing
├── utils/diarization_pipeline.py  # Loads pyannote model, orchestrates diarization
│   └── utils/diarize.py           # Audio preprocessing, speaker segmentation, timestamp alignment
└── utils/result.py                # build_result() — assembles final JSON output
convert_output.py               # Standalone script: JSON → SRT/VTT/TXT formatting
```

**Device handling:** CUDA via `cuda:{device_id}`, Apple Silicon via `mps`. MPS uses `torch.mps.empty_cache()` for memory management.

**Optimization:** Flash Attention 2 (`--flash True`) or SDPA (default). Float16 precision. Batch size defaults to 24; reduce for OOM on MPS (try `--batch-size 4`).

**Diarization** activates when `--hf-token` is provided. Requires a Hugging Face token with access to the pyannote models.

## Code Style

4-space indentation, `snake_case` for functions/variables/modules. No formatter or linter is configured — preserve surrounding style and avoid formatting-only diffs. Keep CLI logic thin; add helpers to `utils/`.

## Commit Conventions

Short imperative subjects with optional prefixes: `feat:`, `doc:`, `docs:`. Keep commits narrowly scoped. PRs should explain user-visible changes and include example CLI output when transcript formatting changes.
