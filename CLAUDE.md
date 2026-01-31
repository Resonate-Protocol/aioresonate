# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
./scripts/setup.sh                           # One-time: create .venv, install deps, pre-commit hooks
./scripts/run-in-env.sh pre-commit run -a    # Run all checks: ruff, mypy, pytest (use before commits)
./scripts/run-in-env.sh pytest               # Run tests only
./scripts/run-in-env.sh pytest tests/server/test_push_stream_behavior.py -v  # Run specific test file
./scripts/run-in-env.sh pytest -k "test_name"  # Run tests matching pattern
```

## Architecture

**aiosendspin** implements the [Sendspin Protocol](https://github.com/Sendspin-Protocol/spec) for synchronized audio streaming across networked devices.

### Core Flow
```
SendspinServer → SendspinGroup → PushStream → Roles → WebSocket clients
```

### Key Components

- **`server/server.py`**: Entry point. Manages mDNS advertising and persistent client registry (client lifetime decoupled from WebSocket connection).

- **`server/group.py`**: Orchestrates synchronized playback. Manages `PushStream`, tracks group state/metadata/volume, coordinates roles via `PlayerGroupRole`.

- **`server/push_stream.py`**: Push-based audio engine. User feeds PCM via `prepare_audio()` → `commit_audio()`. Handles parallel encoding, per-role buffer tracking, backpressure, and late-joiner catch-up caching.

- **`server/roles/`**: Extensible role system. `Role` (ABC) handles per-connection behavior; `GroupRole` (ABC) handles group-level coordination. Built-in: `PlayerRole` (v1), `PlayerGroupRole`.

- **`server/audio_transformers.py`**: Audio encoding/resampling pool with deduplication (FLAC, Opus, PCM variants).

### Audio Pipeline
```
User PCM → prepare_audio() → commit_audio() → parallel encode → check capacity → assign timestamps → route to roles → on_audio_chunk() → WebSocket binary
```

Timestamps are assigned atomically in `commit_audio()` only after all encoding completes.

## Code Style

- Python ≥3.12, fully typed (`mypy --strict`)
- Ruff formatting, 100 char line length
- Conventional commits: `type(scope): subject` (feat, fix, refactor, test, chore)

## Testing

- pytest with `asyncio_mode=auto`
- Tests mirror module structure: `tests/server/`, `tests/models/`, `tests/integration/`
- Snapshots via `syrupy` for protocol message verification
- Common fixtures in `tests/conftest.py`: `pcm_44100_stereo_16bit`, `pcm_48000_stereo_16bit`
