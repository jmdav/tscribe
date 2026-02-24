# tScribe

## Features

- Batch transcribe audio/video files with txt output
- Batch transcript processing using local Ollama models for summarization or grammar
- Review transcripts with diff view / undo for LLM edits

## Requirements

- Python 3.10+ (3.11 recommended)
- `ffmpeg` available on PATH (required by Whisper/faster-whisper)
- Optional: Ollama running locally for transcript processing

## Installation

macOS/Linux:

```bash
chmod +x run.sh
./run.sh
```

Windows (PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./run.ps1
```

By default, put media files in `in/` and check transcripts in `out/`. Processed/edited output goes to `processed/`.