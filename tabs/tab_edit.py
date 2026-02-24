import time
import json
import streamlit as st
import requests
from pathlib import Path

from utils.ollama import get_ollama_models
from utils.text_utils import chunk_text


def render_edit_tab(output_dir: Path, processed_dir: Path):
    st.subheader("AI Grammar & Editing Studio")

    # Initialize processing state
    if "editing" not in st.session_state:
        st.session_state.editing = False

    ollama_models = get_ollama_models()

    if not ollama_models:
        st.warning(
            "⚠️ No local Ollama models detected. Please use the Ollama CLI to install models first."
        )
        return

    if not output_dir.exists():
        st.warning("Raw Output directory does not exist yet.")
        return

    txt_files = list(output_dir.rglob("*.txt"))

    if not txt_files:
        st.info("No raw transcripts available to edit in the Output directory.")
        return

    # --- SETTINGS DROPDOWN ---
    with st.expander("Editing Settings", expanded=False):
        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            selected_model = st.selectbox(
                "Ollama Model",
                ollama_models,
                disabled=st.session_state.editing,
                help="LLM model to use for grammar correction and formatting.",
            )

            # Device selection for Ollama (supports CPU, MPS, and CUDA)
            import torch

            device_options = ["CPU"]
            if torch.backends.mps.is_available():
                device_options.append("MPS")
            if torch.cuda.is_available():
                device_options.append("CUDA")

            device = st.selectbox(
                "Device",
                device_options,
                help="Ollama supports CPU (universal), MPS (Metal GPU on Apple Silicon), and CUDA. The device setting affects inference performance through Ollama's configuration.",
                disabled=st.session_state.editing,
            )

            context_window = st.number_input(
                "Context Window (Tokens)",
                min_value=2048,
                max_value=128000,
                value=16384,
                step=1024,
                disabled=st.session_state.editing,
                help="Maximum tokens the model can process. Larger values allow longer context but use more memory.",
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                disabled=st.session_state.editing,
                help="Randomness of responses (0 = deterministic, 1 = creative). Use 0 for consistent edits.",
            )

        with settings_col2:
            chunk_size = st.slider(
                "Chunk Size (Words)",
                min_value=500,
                max_value=8000,
                value=2500,
                step=500,
                disabled=st.session_state.editing,
                help="Number of words per batch sent to the model. Larger chunks = faster but use more memory.",
            )
            system_prompt = st.text_area(
                "System Prompt",
                value="You are an expert copyeditor. Fix transcription errors, correct the grammar, and format the following lecture transcript. You MUST preserve all original content, concepts, and structure. Do not summarize or remove information. Output only the corrected text.",
                height=120,
                disabled=st.session_state.editing,
                help="Instructions that guide the model's behavior. Be specific about how you want transcripts edited.",
            )

    # --- FILE SELECTION AND PROCESSING ---
    edit_col1, edit_col2 = st.columns([1, 2])

    with edit_col1:
        st.markdown("##### Select Transcripts to Edit")
        all_file_options = [str(f.relative_to(output_dir)) for f in txt_files]

        # Initialize session state if not present
        if "edit_pills" not in st.session_state:
            st.session_state.edit_pills = list(all_file_options)

        sel_col, btn_col, refresh_col = st.columns([2, 2, 1])
        with sel_col:
            if st.button(
                "Select All",
                width="stretch",
                key="select_all_transcripts",
                disabled=st.session_state.editing,
            ):
                st.session_state.edit_pills = list(all_file_options)
                st.rerun()
        with btn_col:
            if st.button(
                "Clear",
                width="stretch",
                key="clear_transcripts",
                disabled=st.session_state.editing,
            ):
                st.session_state.edit_pills = []
                st.rerun()
        with refresh_col:
            if st.button(
                "⟳",
                width="stretch",
                key="refresh_edit",
                disabled=st.session_state.editing,
            ):
                st.rerun()

        target_files = st.pills(
            "Files:",
            all_file_options,
            selection_mode="multi",
            key="edit_pills",
            disabled=st.session_state.editing,
        )

    with edit_col2:
        if target_files:
            st.dataframe(
                [
                    {
                        "Filename": name,
                        "Words": len(
                            (output_dir / name).read_text(encoding="utf-8").split()
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
                disabled=not target_files or st.session_state.editing,
            )
        with btn_col2:
            cancel_btn = st.button(
                "Stop Batch & Save Current",
                type="secondary",
                width="stretch",
                key="stop_batch_edit",
                disabled=not st.session_state.editing,
            )

        if "partial_text" not in st.session_state:
            st.session_state.partial_text = ""
        if "current_file" not in st.session_state:
            st.session_state.current_file = ""

        if cancel_btn:
            st.session_state.cancel_batch = True
            if st.session_state.partial_text and st.session_state.current_file:
                safe_model_name = selected_model.replace(":", "-").replace("/", "_")
                partial_file = (
                    processed_dir
                    / f"{Path(st.session_state.current_file).stem}_{safe_model_name}_PARTIAL.txt"
                )
                partial_file.write_text(st.session_state.partial_text, encoding="utf-8")
                st.warning(f"Batch halted. Progress saved: `{partial_file.name}`")
                st.session_state.partial_text = ""
                st.session_state.current_file = ""
            else:
                st.info("Batch canceled.")
            st.session_state.cancel_batch = False
            st.session_state.editing = False
            st.rerun()

        if process_btn and target_files:
            st.session_state.editing = True
            st.rerun()

    # --- PROCESSING BLOCK ---
    if st.session_state.editing and target_files:
        st.session_state.partial_text = ""
        st.session_state.cancel_batch = False
        safe_model_name = selected_model.replace(":", "-").replace("/", "_")

        batch_progress = st.progress(0.0, text="Batch Progress")

        for file_index, file_str in enumerate(target_files):
            batch_pct = file_index / len(target_files)
            batch_progress.progress(
                batch_pct,
                text=f"File {file_index + 1} of {len(target_files)}",
            )

            if st.session_state.get("cancel_batch", False):
                st.warning("Batch canceled by user.")
                break

            file_path = output_dir / file_str
            st.session_state.current_file = file_path.name

            st.markdown(
                f"### Processing ({file_index + 1}/{len(target_files)}): `{file_path.name}`"
            )

            original_text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(original_text, max_words=chunk_size, context_words=100)
            total_words = max(len(original_text.split()), 1)

            st.info(f"Split into {len(chunks)} chunk(s). (~{total_words} words total)")
            progress_bar = st.progress(0.0, text=f"Edited 0 / {total_words} words")

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

                actual_context = len(user_prompt.split()) + len(system_prompt.split())
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
                        last_ui_update = time.time()
                        for line in res.iter_lines():
                            if line:
                                try:
                                    body = json.loads(line)
                                    if "message" in body and "content" in body.get(
                                        "message", {}
                                    ):
                                        token = body["message"]["content"]
                                        chunk_tokens.append(token)
                                        st.session_state.partial_text += token

                                        now = time.time()
                                        if now - last_ui_update >= 0.3:
                                            last_ui_update = now
                                            current_words = len(
                                                st.session_state.partial_text.split()
                                            )
                                            pct = min(current_words / total_words, 1.0)
                                            progress_bar.progress(
                                                pct,
                                                text=f"Chunk {i+1}/{len(chunks)} · ~{current_words} / {total_words} words",
                                            )
                                            preview = st.session_state.partial_text[
                                                -1500:
                                            ]
                                            if (
                                                len(st.session_state.partial_text)
                                                > 1500
                                            ):
                                                preview = "..." + preview
                                            edited_text_placeholder.markdown(
                                                preview + "▌"
                                            )
                                except json.JSONDecodeError:
                                    continue

                        current_words = len(st.session_state.partial_text.split())
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
                    st.error(f"Error on chunk {i+1}: {type(e).__name__}: {e}")
                    break

            if st.session_state.partial_text:
                new_file_path = (
                    processed_dir / f"{file_path.stem}_{safe_model_name}.txt"
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
        st.session_state.editing = False
        st.rerun()
