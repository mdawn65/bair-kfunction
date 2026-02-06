# Transcription Service (minimal scaffold)

This repository contains a minimal, pluggable transcription service scaffold with:
- FastAPI server (`/submit`, `/status/{job_id}`, `/result/{job_id}`)
- Worker helpers with optional RQ/Redis enqueueing
- Provider adapter pattern (starts with a `dummy` provider)

Quickstart (dev, without Redis):

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the API (dev):

```bash
python main.py
```

3. Submit a file with `curl`:

```bash
curl -F "file=@/path/to/audio.wav" http://localhost:8000/submit
```

4. Check status and result (use returned `job_id`):

```bash
curl http://localhost:8000/status/<job_id>
curl http://localhost:8000/result/<job_id>
```

To enable real providers, set environment variables such as `OPENAI_API_KEY` and implement or enable the corresponding provider adapters in `app/transcribe/providers.py`.
