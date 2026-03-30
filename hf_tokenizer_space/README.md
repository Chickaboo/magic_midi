---
title: Godzilla Piano Tokenizer
emoji: 🎹
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# Godzilla Piano Tokenizer Space

This Space streams the Godzilla piano tar archive, tokenizes each MIDI using a fast `symusic` parser with triplet quantization, and uploads `.npz` outputs to a private Hugging Face dataset repo in resumable batches.

## What It Does

- Downloads source tar from `SOURCE_REPO` / `TAR_FILENAME`
- Builds deterministic member ordering (`sorted(member.name)`) once
- Saves ordering to `metadata/member_index.json` in output dataset repo
- Resumes from `metadata/checkpoint.json` on restart
- Runs integrity check on last 2 completed accepted entries
- Uploads tokenized files in batches (`BATCH_SIZE`, default `500`)
- Writes checkpoint every batch
- Continues safely after crashes/restarts

## Environment Variables (Space Secrets)

Set these in Space Settings -> Variables and secrets:

- `HF_TOKEN` (required): write-enabled Hugging Face token
- `OUTPUT_REPO` (required): target dataset repo, e.g. `username/godzilla-piano-tokenized`
- `SOURCE_REPO` (optional): default `projectlosangeles/Godzilla-MIDI-Dataset`
- `TAR_FILENAME` (optional): default `Godzilla-Piano-MIDI-Dataset-CC-BY-NC-SA.tar.gz`
- `BATCH_SIZE` (optional): default `500`
- `MAX_FILES` (optional): default `0` (0 means no cap)

## Output Dataset Layout

- Token files: `data/{index:07d}_{md5}.npz`
  - `tokens` (`int16`)
  - `onsets` (`float32`)
  - `durations` (`float32`)
- Deterministic index: `metadata/member_index.json`
- Checkpoint: `metadata/checkpoint.json`

## Checkpoint Format

`metadata/checkpoint.json` includes:

```json
{
  "last_completed_index": 4500,
  "accepted": 4387,
  "skipped": 113,
  "last_completed_name": "piano/chopin_op10_1.mid",
  "total_members": 1100000
}
```

Additional internal fields (`recent_completed`, `updated_at`) are used for integrity verification.

## Resume Behavior

On startup:

1. Load `member_index.json` from output repo, or create/upload it if missing.
2. Load `checkpoint.json`.
3. Verify integrity of last two accepted completed entries by downloading `.npz` and checking token lengths.
4. If mismatch, roll back by two entries and reprocess.
5. Resume cleanly from resolved index.

The UI log prints a summary like:

- `Resuming from index 4500 (last: piano/chopin_op10_1.mid)`
- `Integrity check: PASSED`
- `Accepted so far: 4387 | Skipped: 113`

## Running

- Press **Start** to launch the worker thread.
- Press **Stop** to request graceful stop (checkpoint + flush).
- Status/log boxes auto-refresh every 10 seconds.

## Python 3.13 Compatibility

Python 3.13 removes the stdlib `audioop` module. Gradio imports `pydub`, which expects `audioop`.
This Space pins `audioop-lts==0.2.1` in `requirements.txt` for Python 3.13 runtimes.

Python 3.13 also removes `distutils`; older Gradio versions may still import it.
This Space requires `gradio>=5.50.0` to avoid `distutils` imports during startup.

## Notes

- Per-file exceptions are caught and counted as skipped; worker continues.
- Upload operations use exponential backoff retries (5s/15s/30s).
- Tar download retries once after a 60s wait if initial attempts fail.
