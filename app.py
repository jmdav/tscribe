import streamlit as st
import time
from pathlib import Path
from faster_whisper import WhisperModel
import torch

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="tScribe", page_icon="📎", layout="wide")

# Hide Streamlit's default header and footer
hide_st_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            [data-testid="stHeader"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

MODEL_STATS = {
    "tiny.en": "tiny (0.80 acc | 1GB VRAM | 10x Speed)",
    "base.en": "base (0.85 acc | 1.2GB VRAM | 7x Speed)",
    "small.en": "small (0.90 acc | 2GB VRAM | 4x Speed)",
    "medium.en": "medium (0.93 acc | 3.5GB VRAM | 2x Speed)",
    "distil-large-v3": "large (0.95 acc | 4GB VRAM | 6x Speed)",
    "large-v3-turbo": "turbo (0.96 acc | 4GB VRAM | 8x Speed)",
}


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, device: str):
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def find_media_files(directory: Path, exts=None):
    if exts is None:
        exts = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".flac", ".aac", ".ogg"}
    return [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts]


# --- MAIN UI ---
st.markdown("<br>", unsafe_allow_html=True)
st.title("tScribe")

# --- CONFIGURATION ---
with st.expander("Config", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        input_dir_str = st.text_input("Input Directory", value="in")
        model_size = st.selectbox(
            "Model",
            options=list(MODEL_STATS.keys()),
            format_func=lambda x: MODEL_STATS[x],
            index=5,
            help="Select an optimized model based on your available RAM/VRAM.",
        )

    with col2:
        output_dir_str = st.text_input("Output Directory", value="out")
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.insert(0, "cuda")
        device = st.selectbox("Compute Device", device_options)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        enable_vad = st.checkbox(
            "Enable VAD Filter",
            value=True,
            help="Filters out silence before transcribing to drastically improve speed.",
        )
    with col4:
        vad_silence_ms = st.number_input(
            "VAD Min Silence (ms)", value=2000, step=500, disabled=not enable_vad
        )

    st.divider()

input_dir = Path(input_dir_str).expanduser().resolve()
output_dir = Path(output_dir_str).expanduser().resolve()

try:
    output_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


# --- THE FRAGMENT ---
@st.fragment
def run_transcription_engine(in_dir, out_dir, mod_size, dev, use_vad, vad_ms):
    # Add columns to place the Start and Cancel buttons side-by-side
    col1, col2, col3 = st.columns([3, 2, 5])
    with col1:
        start_btn = st.button(
            "Start Batch Transcription", type="primary", use_container_width=True
        )

    if start_btn:
        files = find_media_files(in_dir)
        if not files:
            st.warning(f"No media files found in `{in_dir}`")
            return

        log_container = st.empty()
        logs = []

        def log_msg(msg):
            logs.append(msg)
            log_container.code("\n".join(logs[-12:]), language="text")

        log_msg(f"Loading '{mod_size}' model on {dev}...")
        model = load_model(mod_size, dev)
        log_msg(f"Model loaded. Starting batch processing of {len(files)} file(s).")

        vad_params = dict(min_silence_duration_ms=vad_ms) if use_vad else None
        start_time = time.time()

        for i, src in enumerate(files):
            rel_path = src.relative_to(in_dir)
            out_txt = (out_dir / rel_path).with_suffix(".txt")
            out_txt.parent.mkdir(parents=True, exist_ok=True)

            log_msg(f"Processing ({i+1}/{len(files)}): {src.name}")

            if out_txt.exists():
                log_msg(f"Skipped (already exists): {out_txt.name}")
            else:
                try:
                    segments, info = model.transcribe(
                        str(src),
                        language="en",
                        vad_filter=use_vad,
                        vad_parameters=vad_params,
                        beam_size=5,
                    )

                    # --- LIVE PULSE UI ---
                    file_progress = st.progress(
                        0.0, text=f"Analyzing audio length... ({info.duration:.1f}s)"
                    )
                    live_text_preview = st.empty()

                    text_chunks = []
                    for segment in segments:
                        text_chunks.append(segment.text)

                        percent_done = min(segment.end / info.duration, 1.0)

                        file_progress.progress(
                            percent_done,
                            text=f"Transcribing *{src.name}*: **{percent_done:.0%} ({segment.end:.1f}s / {info.duration:.1f}s)**",
                        )
                        live_text_preview.markdown(f"**Live Preview:** {segment.text}")

                    file_progress.empty()
                    live_text_preview.empty()
                    # ---------------------

                    text = "".join(text_chunks)
                    out_txt.write_text(text.strip(), encoding="utf-8")
                    log_msg(f"Wrote transcript: {out_txt.name}")
                except Exception as e:
                    log_msg(f"! Error on {src.name}: {e}")

        total_time = time.time() - start_time
        m, s = divmod(int(total_time), 60)
        h, m = divmod(m, 60)
        total_str = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

        st.success(f"Batch processing complete in {total_str}!")


tab_transcribe, tab_files, tab_viewer = st.tabs(["Process", "Input", "Output"])

# --- TAB 1: TRANSCRIPTION ---
with tab_transcribe:
    # We call the fragment function here and pass in the current state of our configurations
    run_transcription_engine(
        input_dir, output_dir, model_size, device, enable_vad, vad_silence_ms
    )

# --- TAB 2: INPUT EXPLORER ---
with tab_files:
    if input_dir.exists():
        media_files = find_media_files(input_dir)
        if media_files:
            st.dataframe(
                [
                    {
                        "Filename": f.name,
                        "Size (MB)": round(f.stat().st_size / 1048576, 2),
                    }
                    for f in media_files
                ],
                width="content",
            )
        else:
            st.info("No media files found in the input directory.")
    else:
        st.error(f"Directory not found: `{input_dir}`")

# --- TAB 3: TRANSCRIPT VIEWER ---
with tab_viewer:
    if output_dir.exists():
        txt_files = list(output_dir.rglob("*.txt"))
        if txt_files:
            selected = st.selectbox(
                "Select a transcript to preview:",
                [str(f.relative_to(output_dir)) for f in txt_files],
            )
            if selected:
                file_path = output_dir / selected
                content = file_path.read_text(encoding="utf-8")

                st.download_button(
                    label="Download .txt",
                    data=content,
                    file_name=file_path.name,
                    mime="text/plain",
                )
                st.text_area("Content", content, height=400, disabled=False)
        else:
            st.info("No transcripts found yet. Run the transcription engine first!")
