import argparse
from pathlib import Path
import whisper
import sys


def find_media_files(directory: Path, exts=None):
    if exts is None:
        exts = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".flac", ".aac", ".ogg"}
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def transcribe_dir(
    input_dir: Path,
    output_dir: Path,
    model_name: str = "turbo",
    overwrite: bool = False,
    language: str = None,
):
    # Accept either `str` or `Path` for input/output and coerce to Path
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name, device="mps")

    files = list(find_media_files(input_dir))
    if not files:
        print(f"No supported media files found in {input_dir}")
        return

    print(f"Found {len(files)} file(s). Beginning transcription...")
    for src in files:
        rel = src.relative_to(input_dir)
        out_base = (output_dir / rel).with_suffix("")
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_txt = out_base.with_suffix(".txt")

        if out_txt.exists() and not overwrite:
            print(f"Skipping (exists): {out_txt}")
            continue

        try:
            print(f"Transcribing: {src}")
            result = (
                model.transcribe(str(src), language=language)
                if language
                else model.transcribe(str(src))
            )
            text = result.get("text", "").strip()
            out_txt.write_text(text, encoding="utf-8")
            print(f"Wrote: {out_txt}")
        except Exception as e:
            print(f"Error transcribing {src}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe all media files in a directory to text files"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="in",
        help="Input directory (default: `in`)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="out",
        help="Output directory for transcripts (default: `out`)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="turbo",
        help="Whisper model to use (tiny, base, small, medium, large)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing transcript files"
    )
    parser.add_argument(
        "--language", default=None, help='Optional language hint (e.g., "en")'
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else Path("out")

    # Create default input/output directories if they don't exist
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    transcribe_dir(
        input_dir,
        output_dir,
        model_name=args.model,
        overwrite=args.overwrite,
        language=args.language,
    )


if __name__ == "__main__":
    main()
