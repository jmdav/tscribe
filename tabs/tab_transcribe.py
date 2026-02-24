import time
import streamlit as st
import torch
from pathlib import Path

from config import MODEL_STATS
from utils.file_utils import find_media_files, get_media_duration
from utils.model import load_model


def render_transcribe_tab(in_dir: Path, out_dir: Path):
    """Transcription tab with built-in settings and file selection."""

    # Initialize processing state
    if "transcribing" not in st.session_state:
        st.session_state.transcribing = False

    # --- SETTINGS DROPDOWN ---
    with st.expander("Transcription Settings", expanded=False):
        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            model_size = st.selectbox(
                "Model",
                options=list(MODEL_STATS.keys()),
                format_func=lambda x: MODEL_STATS[x],
                index=5,
                help="Select an optimized model based on your available RAM/VRAM.",
                disabled=st.session_state.transcribing,
            )

        with settings_col2:
            # Note: faster_whisper does NOT support MPS, only CPU and CUDA
            device_options = ["cpu"]
            if torch.cuda.is_available():
                device_options.insert(0, "cuda")

            device = st.selectbox(
                "Device",
                device_options,
                disabled=st.session_state.transcribing,
                help="faster_whisper supports CPU and CUDA only (no MPS)",
            )

        vad_col1, vad_col2 = st.columns(2)
        with vad_col1:
            enable_vad = st.selectbox(
                "VAD Filter",
                options=[True, False],
                index=0,
                format_func=lambda x: "Enabled" if x else "Disabled",
                help="Filters out silence before transcribing.",
                disabled=st.session_state.transcribing,
            )

        with vad_col2:
            vad_silence_ms = st.number_input(
                "VAD Min Silence (ms)",
                value=2000,
                step=500,
                disabled=not enable_vad or st.session_state.transcribing,
                help="Minimum silence duration (in milliseconds) to trigger pause detection.",
            )

    # --- FILE SELECTION AND PROCESSING ---
    media_files = find_media_files(in_dir)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### Select Files to Transcribe")
        if media_files:
            all_media_names = [f.name for f in media_files]

            # Initialize session state if not present
            if "media_pills" not in st.session_state:
                st.session_state.media_pills = list(all_media_names)

            sel_col, btn_col, refresh_col = st.columns([2, 2, 1])
            with sel_col:
                if st.button(
                    "Select All",
                    width="stretch",
                    key="select_all_media",
                    disabled=st.session_state.transcribing,
                ):
                    st.session_state.media_pills = list(all_media_names)
                    st.rerun()
            with btn_col:
                if st.button(
                    "Clear",
                    width="stretch",
                    key="clear_media",
                    disabled=st.session_state.transcribing,
                ):
                    st.session_state.media_pills = []
                    st.rerun()
            with refresh_col:
                if st.button(
                    "⟳",
                    width="stretch",
                    key="refresh_media",
                    disabled=st.session_state.transcribing,
                ):
                    st.rerun()

            selected_media = st.pills(
                "Files:",
                all_media_names,
                selection_mode="multi",
                key="media_pills",
                disabled=st.session_state.transcribing,
            )
        else:
            selected_media = []
            st.info("No media files found in input directory.")

        # Build lookup for selected files
        media_lookup = {f.name: f for f in media_files}
        selected_paths = [
            media_lookup[name] for name in selected_media if name in media_lookup
        ]

        start_btn = st.button(
            (
                f"Transcribe {len(selected_paths)} File(s)"
                if selected_paths
                else "No Files Selected"
            ),
            type="primary",
            width="stretch",
            disabled=not selected_paths or st.session_state.transcribing,
        )

    with col2:
        if selected_paths:
            st.markdown("**Selected Files:**")
            summary_data = [
                {
                    "Filename": f.name,
                    "Type": f.suffix.upper().lstrip("."),
                    "Length": get_media_duration(f),
                    "Size (MB)": round(f.stat().st_size / 1048576, 2),
                }
                for f in selected_paths
            ]
            st.dataframe(summary_data, hide_index=True, width="stretch")

        if start_btn and selected_paths:
            st.session_state.transcribing = True
            st.rerun()

    # --- PROCESSING BLOCK ---
    if st.session_state.transcribing and selected_paths:
        log_container = st.empty()
        logs = []

        def log_msg(msg):
            logs.append(msg)
            log_container.code("\n".join(logs[-12:]), language="text")

        log_msg(f"Loading '{model_size}' model on {device}...")
        model = load_model(model_size, device)
        log_msg(
            f"Model loaded. Starting batch processing of {len(selected_paths)} file(s)."
        )

        vad_params = (
            dict(min_silence_duration_ms=vad_silence_ms) if enable_vad else None
        )
        start_time = time.time()

        for i, src in enumerate(selected_paths):
            rel_path = src.relative_to(in_dir)
            out_txt = (out_dir / rel_path).with_suffix(".txt")
            out_txt.parent.mkdir(parents=True, exist_ok=True)

            log_msg(f"Processing ({i+1}/{len(selected_paths)}): {src.name}")

            if out_txt.exists():
                log_msg(f"Skipped (already exists): {out_txt.name}")
            else:
                try:
                    segments, info = model.transcribe(
                        str(src),
                        language="en",
                        vad_filter=enable_vad,
                        vad_parameters=vad_params,
                        beam_size=5,
                    )
                except Exception as e:
                    log_msg(
                        f"! Transcription failed on {src.name}: {type(e).__name__}: {e}"
                    )
                    continue

                file_progress = st.progress(
                    0.0, text=f"Processing {info.duration:.1f}s audio..."
                )
                with st.container(height=350, border=True):
                    live_preview = st.empty()

                text_chunks = []
                segment_count = 0
                try:
                    for segment in segments:
                        text_chunks.append(segment.text)
                        segment_count += 1

                        if segment_count % 5 == 0 or segment.end >= info.duration:
                            percent_done = min(segment.end / info.duration, 1.0)
                            file_progress.progress(
                                percent_done,
                                text=f"Transcribing *{src.name}*: {percent_done:.0%}",
                            )
                            full_text = " ".join(text_chunks).strip()
                            live_preview.markdown(full_text + "▌")

                    file_progress.empty()
                    live_preview.empty()

                    text = " ".join(text_chunks).strip()
                    out_txt.write_text(text, encoding="utf-8")
                    log_msg(f"Wrote transcript: {out_txt.name} ({len(text)} chars)")
                except Exception as e:
                    log_msg(f"! Processing error on {src.name}: {type(e).__name__}")
                    file_progress.empty()
                    live_preview.empty()

        total_time = time.time() - start_time
        m, s = divmod(int(total_time), 60)
        h, m = divmod(m, 60)
        total_str = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m:02d}m {s:02d}s"

        st.success(f"Batch processing complete in {total_str}!")
        st.session_state.transcribing = False
        st.rerun()
