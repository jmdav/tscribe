import streamlit as st
import time
from pathlib import Path
from faster_whisper import WhisperModel
import torch
import requests
import re

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="tScribe", page_icon="📎", layout="wide")

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


# --- NEW: OLLAMA HELPERS ---
def get_ollama_models():
    """Fetches installed models from the local Ollama instance."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
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
    col1, col2, col3 = st.columns(3)

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
        output_dir_str = st.text_input("Raw Output Directory", value="out")
        device_options = ["cpu"]
        if torch.cuda.is_available():
            device_options.insert(0, "cuda")
        device = st.selectbox("Compute Device", device_options)

    with col3:
        # NEW: Processed directory input
        processed_dir_str = st.text_input("Processed Directory", value="processed")

    st.divider()

    col_vad1, col_vad2 = st.columns(2)
    with col_vad1:
        enable_vad = st.checkbox(
            "Enable VAD Filter",
            value=True,
            help="Filters out silence before transcribing to drastically improve speed.",
        )
    with col_vad2:
        vad_silence_ms = st.number_input(
            "VAD Min Silence (ms)", value=2000, step=500, disabled=not enable_vad
        )

    st.divider()

input_dir = Path(input_dir_str).expanduser().resolve()
output_dir = Path(output_dir_str).expanduser().resolve()
processed_dir = Path(processed_dir_str).expanduser().resolve()  # NEW

# Ensure all directories exist
for d in [input_dir, output_dir, processed_dir]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# --- THE FRAGMENT ---
@st.fragment
def run_transcription_engine(in_dir, out_dir, mod_size, dev, use_vad, vad_ms):
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


# Note the new tab for post-processing
tab_transcribe, tab_files, tab_viewer, tab_post = st.tabs(
    ["Process", "Input", "Output", "Post-Process (Ollama)"]
)

# --- TAB 1: TRANSCRIPTION ---
with tab_transcribe:
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

# --- TAB 4: OLLAMA BATCH POST-PROCESSING ---
with tab_post:
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
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("##### Batch Setup")
                    # NEW: Multiselect allows batching multiple files at once
                    target_files = st.multiselect(
                        "Select Transcripts to Edit:",
                        [str(f.relative_to(output_dir)) for f in txt_files],
                        default=[str(f.relative_to(output_dir)) for f in txt_files],
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

                    # UPDATED: Now uses word counts to match the regex chunker
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

                with col2:
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        process_btn = st.button(
                            "Start Batch Editing",
                            type="primary",
                            use_container_width=True,
                            key="start_batch_edit",
                        )
                    with btn_col2:
                        cancel_btn = st.button(
                            "🛑 Stop Batch & Save Current",
                            type="secondary",
                            use_container_width=True,
                            key="stop_batch_edit",
                        )

                    if "partial_text" not in st.session_state:
                        st.session_state.partial_text = ""
                    if "current_file" not in st.session_state:
                        st.session_state.current_file = ""

                    if cancel_btn:
                        if (
                            st.session_state.partial_text
                            and st.session_state.current_file
                        ):
                            safe_model_name = selected_model.replace(":", "-").replace(
                                "/", "_"
                            )
                            # Save rescues to the new processed_dir
                            partial_file = (
                                processed_dir
                                / f"{Path(st.session_state.current_file).stem}_{safe_model_name}_PARTIAL.txt"
                            )
                            partial_file.write_text(
                                st.session_state.partial_text, encoding="utf-8"
                            )
                            st.warning(
                                f"Batch halted. Rescued progress saved to Processed folder as `{partial_file.name}`."
                            )
                            st.session_state.partial_text = ""
                            st.session_state.current_file = ""
                        else:
                            st.info("Batch canceled.")

                    if process_btn and target_files:
                        import json

                        st.session_state.partial_text = ""
                        safe_model_name = selected_model.replace(":", "-").replace(
                            "/", "_"
                        )

                        # NEW: Outer loop iterating through all selected files
                        for file_index, file_str in enumerate(target_files):
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
                                if context_text:
                                    user_prompt = (
                                        f"For grammatical continuity, here is the end of the previous section. "
                                        f"DO NOT EDIT OR OUTPUT THIS CONTEXT:\n[CONTEXT START]\n{context_text}\n[CONTEXT END]\n\n"
                                        f"---\n\n"
                                        f"TEXT TO EDIT:\n{chunk}"
                                    )
                                else:
                                    user_prompt = f"TEXT TO EDIT:\n{chunk}"

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
                                        "num_ctx": context_window,
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

                                        for line in res.iter_lines():
                                            if line:
                                                body = json.loads(line)
                                                if (
                                                    "message" in body
                                                    and "content" in body["message"]
                                                ):
                                                    token = body["message"]["content"]

                                                    st.session_state.partial_text += (
                                                        token
                                                    )

                                                    current_words = len(
                                                        st.session_state.partial_text.split()
                                                    )
                                                    pct = min(
                                                        current_words / total_words, 1.0
                                                    )
                                                    progress_bar.progress(
                                                        pct,
                                                        text=f"Edited ~{current_words} / {total_words} words",
                                                    )

                                                    edited_text_placeholder.markdown(
                                                        st.session_state.partial_text
                                                        + "▌"
                                                    )

                                        edited_text_placeholder.markdown(
                                            st.session_state.partial_text
                                        )
                                        st.session_state.partial_text += "\n\n"

                                except Exception as e:
                                    st.error(f"Ollama API Error on chunk {i+1}: {e}")
                                    break

                            if st.session_state.partial_text:
                                # NEW: Save completed files to the processed_dir
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
