import subprocess
from pathlib import Path


def find_media_files(directory: Path, exts=None):
    if exts is None:
        exts = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".flac", ".aac", ".ogg"}
    if not directory.exists():
        return []
    try:
        return sorted(
            [
                p
                for p in directory.rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            ]
        )
    except PermissionError:
        return []


def get_media_duration(filepath: Path) -> str:
    """Get duration of a media file using ffprobe. Returns formatted string like '1:23:45'."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        seconds = float(result.stdout.strip())
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    except Exception:
        return "—"


def validate_directory(directory: Path, create: bool = True) -> bool:
    """Validate and optionally create directory. Returns True if accessible."""
    try:
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory.exists() and directory.is_dir()
    except (PermissionError, OSError):
        return False
