import streamlit as st
import time
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel
import torch
import requests
import re
import json
import logging

logging.basicConfig(level=logging.WARNING)

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="tScribe", page_icon="📎", layout="wide")

hide_st_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            [data-testid="stHeader"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            .stMarkdown h1 a, .stMarkdown h2 a, .stMarkdown h3 a,
            .stMarkdown h4 a, .stMarkdown h5 a, .stMarkdown h6 a,
            [data-testid="stHeaderActionElements"] {display: none !important;}
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


@st.cache_resource(show_spinner=True)
def load_model(model_size: str, device: str):
    try:
        if device == "cuda":
            compute_type = "float16"
        elif device == "mps":
            # MPS doesn't support int8; use float16
            compute_type = "float16"
        else:
            compute_type = "int8"
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        st.error(f"Failed to load model '{model_size}': {e}")
        raise


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


# --- OLLAMA HELPERS ---
@st.cache_data(ttl=30, show_spinner=False)
def get_ollama_models():
    """Fetches installed models from the local Ollama instance. Cached for 30s."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except (requests.ConnectionError, requests.Timeout):
        return []
    except Exception as e:
        logging.warning(f"Failed to fetch Ollama models: {e}")
        return []


def chunk_text(text, max_words=2000, context_words=100):
    """
    Splits text safely by sentence boundaries.
    Returns tuples of: (context_from_previous_chunk, text_to_edit)
    """
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    protected_text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|vs)\.\s+", r"\1<DOT> ", text)
    raw_sentences = re.split(sentence_pattern, protected_text)

    raw_chunks = []
    current_chunk = []
    current_word_count = 0

    # 1. Build the chunks using word counts
    for sentence in raw_sentences:
        sentence = sentence.replace("<DOT>", ".")
        sentence_word_count = len(sentence.split())

        if sentence_word_count > max_words:
            if current_chunk:
                raw_chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            raw_chunks.append(sentence)
            continue

        if current_word_count + sentence_word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            raw_chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count

    if current_chunk:
        raw_chunks.append(" ".join(current_chunk))

    # 2. Attach the sliding window context
    chunks_with_context = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            # First chunk has no previous context
            chunks_with_context.append(("", chunk))
        else:
            # Grab the last N words from the previous chunk
            prev_chunk_words = raw_chunks[i - 1].split()
            context = " ".join(prev_chunk_words[-context_words:])
            chunks_with_context.append((context, chunk))

    return chunks_with_context


# --- MAIN UI ---
st.markdown("<br>", unsafe_allow_html=True)
st.title("tScribe")

# --- CONFIGURATION ---
with st.expander("Config", expanded=True):
    col_left, col_right = st.columns(2)

    with col_left:
        st.caption("DIRECTORIES")
        input_dir_str = st.text_input("Input", value="in")
        output_dir_str = st.text_input("Output", value="out")
        processed_dir_str = st.text_input("Processed", value="processed")

    with col_right:
        st.caption("TRANSCRIPTION")
        model_size = st.selectbox(
            "Model",
            options=list(MODEL_STATS.keys()),
            format_func=lambda x: MODEL_STATS[x],
            index=5,
            help="Select an optimized model based on your available RAM/VRAM.",
        )
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.insert(0, "cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_options.insert(0, "mps")
        dev_col, vad_col = st.columns(2)
        with dev_col:
            device = st.selectbox("Device", device_options)
        with vad_col:
            enable_vad = st.selectbox(
                "VAD Filter",
                options=[True, False],
                index=0,
                format_func=lambda x: "Enabled" if x else "Disabled",
                help="Filters out silence before transcribing.",
            )
        if enable_vad:
            vad_silence_ms = st.number_input(
                "VAD Min Silence (ms)", value=2000, step=500
            )
        else:
            vad_silence_ms = 2000

input_dir = Path(input_dir_str).expanduser().resolve()
output_dir = Path(output_dir_str).expanduser().resolve()
processed_dir = Path(processed_dir_str).expanduser().resolve()

# Validate all directories
errors = []
if not validate_directory(input_dir, create=True):
    errors.append(f"Cannot access input directory: {input_dir}")
if not validate_directory(output_dir, create=True):
    errors.append(f"Cannot access output directory: {output_dir}")
if not validate_directory(processed_dir, create=True):
    errors.append(f"Cannot access processed directory: {processed_dir}")

if errors:
    for error in errors:
        st.error(error)


# --- THE FRAGMENT ---
@st.fragment
def run_transcription_engine(in_dir, out_dir, proc_dir, mod_size, dev, use_vad, vad_ms):
    media_files = find_media_files(in_dir)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("##### Select Files to Transcribe")
        if media_files:
            all_media_names = [f.name for f in media_files]

            sel_col, btn_col = st.columns([1, 1])
            with sel_col:
                if st.button("Select All", width="stretch", key="select_all_media"):
                    st.session_state.media_pills = list(all_media_names)
                    st.rerun()
            with btn_col:
                if st.button("Clear", width="stretch", key="clear_media"):
                    st.session_state.media_pills = []
                    st.rerun()

            selected_media = st.pills(
                "Files:",
                all_media_names,
                default=st.session_state.get("media_pills", list(all_media_names)),
                selection_mode="multi",
                key="media_pills",
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
            disabled=not selected_paths,
        )

    with col2:
        # Show summary of selected files
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
            st.dataframe(summary_data, hide_index=True, use_container_width=True)

        if start_btn and selected_paths:
            log_container = st.empty()
            logs = []

            def log_msg(msg):
                logs.append(msg)
                log_container.code("\n".join(logs[-12:]), language="text")

            log_msg(f"Loading '{mod_size}' model on {dev}...")
            model = load_model(mod_size, dev)
            log_msg(
                f"Model loaded. Starting batch processing of {len(selected_paths)} file(s)."
            )

            vad_params = dict(min_silence_duration_ms=vad_ms) if use_vad else None
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
                            vad_filter=use_vad,
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
                                live_preview.caption(
                                    f"...{segment.text.strip()[-120:]}"
                                )

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
            st.rerun()


tab_transcribe, tab_viewer, tab_edit = st.tabs(["Transcribe", "Transcripts", "Edit"])

# --- TAB 1: TRANSCRIPTION ---
with tab_transcribe:
    run_transcription_engine(
        input_dir,
        output_dir,
        processed_dir,
        model_size,
        device,
        enable_vad,
        vad_silence_ms,
    )

# --- TAB 2: TRANSCRIPT VIEWER ---
with tab_viewer:
    viewer_sources = {}
    if output_dir.exists():
        for f in sorted(output_dir.rglob("*.txt")):
            viewer_sources[f"[raw] {f.relative_to(output_dir)}"] = f
    if processed_dir.exists():
        for f in sorted(processed_dir.rglob("*.txt")):
            viewer_sources[f"[edited] {f.relative_to(processed_dir)}"] = f

    if viewer_sources:
        viewer_col1, viewer_col2 = st.columns([1, 2])

        with viewer_col1:
            selected_transcript = st.selectbox(
                "Select a transcript:",
                list(viewer_sources.keys()),
            )
            if selected_transcript:
                file_path = viewer_sources[selected_transcript]
                content = file_path.read_text(encoding="utf-8")
                word_count = len(content.split())

                st.caption(f"{word_count:,} words · {len(content):,} chars")
                st.download_button(
                    label="Download .txt",
                    data=content,
                    file_name=file_path.name,
                    mime="text/plain",
                    width="stretch",
                )

        with viewer_col2:
            if selected_transcript:
                st.text_area(
                    "Content",
                    content,
                    height=500,
                    disabled=False,
                    label_visibility="collapsed",
                )
    else:
        st.info("No transcripts found yet. Run the transcription engine first!")

# --- TAB 3: OLLAMA BATCH POST-PROCESSING ---
with tab_edit:
    st.subheader("AI Grammar & Editing Studio")

    ollama_models = get_ollama_models()

    if not ollama_models:
        st.warning(
            "⚠️ No local Ollama models detected. Please use the Ollama CLI to install models first."
        )
    else:
        if output_dir.exists():
            txt_files = list(output_dir.rglob("*.txt"))

            if txt_files:
                edit_col1, edit_col2 = st.columns([1, 2])

                with edit_col1:
                    st.markdown("##### Select Transcripts to Edit")
                    all_file_options = [
                        str(f.relative_to(output_dir)) for f in txt_files
                    ]

                    sel_col, btn_col = st.columns([1, 1])
                    with sel_col:
                        if st.button(
                            "Select All",
                            width="stretch",
                            key="select_all_transcripts",
                        ):
                            st.session_state.edit_pills = list(all_file_options)
                            st.rerun()
                    with btn_col:
                        if st.button("Clear", width="stretch", key="clear_transcripts"):
                            st.session_state.edit_pills = []
                            st.rerun()

                    target_files = st.pills(
                        "Files:",
                        all_file_options,
                        default=st.session_state.get(
                            "edit_pills", list(all_file_options)
                        ),
                        selection_mode="multi",
                        key="edit_pills",
                    )

                    selected_model = st.selectbox("Select Ollama Model:", ollama_models)

                    st.markdown("##### Inference Settings")
                    context_window = st.number_input(
                        "Context Window (Tokens)",
                        min_value=2048,
                        max_value=128000,
                        value=16384,
                        step=1024,
                    )
                    temperature = st.slider(
                        "Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1
                    )

                    chunk_size = st.slider(
                        "Chunk Size (Words)",
                        min_value=500,
                        max_value=8000,
                        value=2500,
                        step=500,
                    )

                    system_prompt = st.text_area(
                        "System Prompt:",
                        value="You are an expert copyeditor. Fix transcription errors, correct the grammar, and format the following lecture transcript. You MUST preserve all original content, concepts, and structure. Do not summarize or remove information. Output only the corrected text.",
                    )

                with edit_col2:
                    # Show details table for selected files
                    if target_files:
                        st.dataframe(
                            [
                                {
                                    "Filename": name,
                                    "Words": len(
                                        (output_dir / name)
                                        .read_text(encoding="utf-8")
                                        .split()
                                    ),
                                    "Size (KB)": round(
                                        (output_dir / name).stat().st_size / 1024, 1
                                    ),
                                }
                                for name in target_files
                                if (output_dir / name).exists()
                            ],
                            width="stretch",
                            hide_index=True,
                        )

                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        process_btn = st.button(
                            "Start Batch Editing",
                            type="primary",
                            width="stretch",
                            key="start_batch_edit",
                        )
                    with btn_col2:
                        cancel_btn = st.button(
                            "🛑 Stop Batch & Save Current",
                            type="secondary",
                            width="stretch",
                            key="stop_batch_edit",
                        )

                    if "partial_text" not in st.session_state:
                        st.session_state.partial_text = ""
                    if "current_file" not in st.session_state:
                        st.session_state.current_file = ""

                    if cancel_btn:
                        st.session_state.cancel_batch = True
                        if (
                            st.session_state.partial_text
                            and st.session_state.current_file
                        ):
                            safe_model_name = selected_model.replace(":", "-").replace(
                                "/", "_"
                            )
                            partial_file = (
                                processed_dir
                                / f"{Path(st.session_state.current_file).stem}_{safe_model_name}_PARTIAL.txt"
                            )
                            partial_file.write_text(
                                st.session_state.partial_text, encoding="utf-8"
                            )
                            st.warning(
                                f"Batch halted. Progress saved: `{partial_file.name}`"
                            )
                            st.session_state.partial_text = ""
                            st.session_state.current_file = ""
                        else:
                            st.info("Batch canceled.")
                        st.session_state.cancel_batch = False

                    if process_btn and target_files:
                        st.session_state.partial_text = ""
                        st.session_state.cancel_batch = False
                        safe_model_name = selected_model.replace(":", "-").replace(
                            "/", "_"
                        )

                        # Outer loop iterating through all selected files
                        for file_index, file_str in enumerate(target_files):
                            if st.session_state.get("cancel_batch", False):
                                st.warning("Batch canceled by user.")
                                break
                            file_path = output_dir / file_str
                            st.session_state.current_file = file_path.name

                            st.markdown(
                                f"### Processing ({file_index + 1}/{len(target_files)}): `{file_path.name}`"
                            )

                            original_text = file_path.read_text(encoding="utf-8")
                            chunks = chunk_text(
                                original_text, max_words=chunk_size, context_words=100
                            )
                            total_words = max(len(original_text.split()), 1)

                            st.info(
                                f"Split into {len(chunks)} chunk(s). (~{total_words} words total)"
                            )
                            progress_bar = st.progress(
                                0.0, text=f"Edited 0 / {total_words} words"
                            )

                            with st.container(height=350, border=True):
                                edited_text_placeholder = st.empty()

                            for i, (context_text, chunk) in enumerate(chunks):
                                if st.session_state.get("cancel_batch", False):
                                    break

                                if context_text:
                                    user_prompt = (
                                        f"For grammatical continuity, here is the end of the previous section. "
                                        f"DO NOT EDIT OR OUTPUT THIS CONTEXT:\n[CONTEXT START]\n{context_text}\n[CONTEXT END]\n\n"
                                        f"---\n\n"
                                        f"TEXT TO EDIT:\n{chunk}"
                                    )
                                else:
                                    user_prompt = f"TEXT TO EDIT:\n{chunk}"

                                # Validate context window isn't too small
                                actual_context = len(user_prompt.split()) + len(
                                    system_prompt.split()
                                )
                                if context_window < actual_context + 100:
                                    st.warning(
                                        f"Context window ({context_window}) may be insufficient. Increasing to {actual_context + 500}."
                                    )
                                    effective_context = actual_context + 500
                                else:
                                    effective_context = context_window

                                payload = {
                                    "model": selected_model,
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt},
                                    ],
                                    "stream": True,
                                    "options": {
                                        "temperature": temperature,
                                        "num_predict": -1,
                                        "num_ctx": effective_context,
                                    },
                                }

                                try:
                                    with st.spinner(f"Chunk {i+1} of {len(chunks)}..."):
                                        res = requests.post(
                                            "http://localhost:11434/api/chat",
                                            json=payload,
                                            stream=True,
                                            timeout=1800,
                                        )
                                        res.raise_for_status()

                                        chunk_tokens = []
                                        token_count = 0
                                        last_ui_update = time.time()
                                        for line in res.iter_lines():
                                            if line:
                                                try:
                                                    body = json.loads(line)
                                                    if (
                                                        "message" in body
                                                        and "content"
                                                        in body.get("message", {})
                                                    ):
                                                        token = body["message"][
                                                            "content"
                                                        ]
                                                        chunk_tokens.append(token)
                                                        st.session_state.partial_text += (
                                                            token
                                                        )
                                                        token_count += 1

                                                        # Live update every 0.3s for responsiveness
                                                        now = time.time()
                                                        if now - last_ui_update >= 0.3:
                                                            last_ui_update = now
                                                            current_words = len(
                                                                st.session_state.partial_text.split()
                                                            )
                                                            pct = min(
                                                                current_words
                                                                / total_words,
                                                                1.0,
                                                            )
                                                            progress_bar.progress(
                                                                pct,
                                                                text=f"Chunk {i+1}/{len(chunks)} · ~{current_words} / {total_words} words",
                                                            )
                                                            # Show tail of generated text for live feel
                                                            preview = st.session_state.partial_text[
                                                                -1500:
                                                            ]
                                                            if (
                                                                len(
                                                                    st.session_state.partial_text
                                                                )
                                                                > 1500
                                                            ):
                                                                preview = (
                                                                    "..." + preview
                                                                )
                                                            edited_text_placeholder.markdown(
                                                                preview + "▌"
                                                            )
                                                except json.JSONDecodeError:
                                                    continue

                                        # Final update for this chunk
                                        current_words = len(
                                            st.session_state.partial_text.split()
                                        )
                                        pct = min(current_words / total_words, 1.0)
                                        progress_bar.progress(
                                            pct,
                                            text=f"Edited ~{current_words} / {total_words} words",
                                        )
                                        preview = st.session_state.partial_text[-1500:]
                                        if len(st.session_state.partial_text) > 1500:
                                            preview = "..." + preview
                                        edited_text_placeholder.markdown(preview)
                                        st.session_state.partial_text += "\n\n"

                                except requests.exceptions.Timeout:
                                    st.error(
                                        f"Timeout on chunk {i+1}: Ollama took too long. Check if model is running."
                                    )
                                    break
                                except requests.exceptions.ConnectionError:
                                    st.error(
                                        f"Connection error on chunk {i+1}: Cannot reach Ollama at localhost:11434. Is it running?"
                                    )
                                    break
                                except Exception as e:
                                    st.error(
                                        f"Error on chunk {i+1}: {type(e).__name__}: {e}"
                                    )
                                    break

                            if st.session_state.partial_text:
                                # Save completed files to the processed_dir
                                new_file_path = (
                                    processed_dir
                                    / f"{file_path.stem}_{safe_model_name}.txt"
                                )
                                new_file_path.write_text(
                                    st.session_state.partial_text, encoding="utf-8"
                                )
                                st.success(
                                    f"File complete! Saved to `{new_file_path.name}` in Processed folder."
                                )
                                st.session_state.partial_text = ""
                                st.session_state.current_file = ""

                        st.balloons()
            else:
                st.info("No raw transcripts available to edit in the Output directory.")
        else:
            st.warning("Raw Output directory does not exist yet.")
