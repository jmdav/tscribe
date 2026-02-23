import re
import difflib
import html as html_mod


def generate_html_diff(raw_text: str, edited_text: str) -> str:
    """
    Generate an HTML diff view highlighting changes between raw and edited text.
    Uses word-level diffing for better readability.
    """
    raw_words = raw_text.split()
    edited_words = edited_text.split()

    diff = difflib.SequenceMatcher(None, raw_words, edited_words)
    html_parts = []

    for tag, i1, i2, j1, j2 in diff.get_opcodes():
        if tag == "equal":
            html_parts.append(" ".join(raw_words[i1:i2]))
        elif tag == "delete":
            deleted = " ".join(raw_words[i1:i2])
            html_parts.append(
                f'<span style="background-color: #ffcccc; text-decoration: line-through;">{deleted}</span>'
            )
        elif tag == "insert":
            inserted = " ".join(edited_words[j1:j2])
            html_parts.append(
                f'<span style="background-color: #ccffcc; font-weight: bold;">{inserted}</span>'
            )
        elif tag == "replace":
            deleted = " ".join(raw_words[i1:i2])
            inserted = " ".join(edited_words[j1:j2])
            html_parts.append(
                f'<span style="background-color: #ffcccc; text-decoration: line-through;">{deleted}</span> '
            )
            html_parts.append(
                f'<span style="background-color: #ccffcc; font-weight: bold;">{inserted}</span>'
            )

        html_parts.append(" ")

    return " ".join(html_parts)


def build_paragraph_diffs(raw_text: str, edited_text: str):
    """
    Build paragraph-aligned diff data. Returns a list of dicts, one per paragraph,
    each containing:
      - 'diff_html': the paragraph text with inline diff highlighting
      - 'changes': list of individual changes [{id, old, new, tag}]
    """
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
    edited_paras = [p.strip() for p in re.split(r"\n\s*\n", edited_text) if p.strip()]

    para_matcher = difflib.SequenceMatcher(None, raw_paras, edited_paras)
    result = []
    change_id = 0

    for tag, i1, i2, j1, j2 in para_matcher.get_opcodes():
        if tag == "equal":
            for idx in range(i1, i2):
                para_result = _diff_paragraph(
                    raw_paras[idx], edited_paras[j1 + (idx - i1)], change_id
                )
                change_id = para_result["next_id"]
                result.append(para_result)
        elif tag == "replace":
            raw_slice = raw_paras[i1:i2]
            edit_slice = edited_paras[j1:j2]
            max_len = max(len(raw_slice), len(edit_slice))
            for k in range(max_len):
                rp = raw_slice[k] if k < len(raw_slice) else ""
                ep = edit_slice[k] if k < len(edit_slice) else ""
                para_result = _diff_paragraph(rp, ep, change_id)
                change_id = para_result["next_id"]
                result.append(para_result)
        elif tag == "delete":
            for idx in range(i1, i2):
                para_result = _diff_paragraph(raw_paras[idx], "", change_id)
                change_id = para_result["next_id"]
                result.append(para_result)
        elif tag == "insert":
            for idx in range(j1, j2):
                para_result = _diff_paragraph("", edited_paras[idx], change_id)
                change_id = para_result["next_id"]
                result.append(para_result)

    return result


def _diff_paragraph(raw_para: str, edited_para: str, start_id: int) -> dict:
    """
    Diff a single paragraph at the word level.
    Returns dict with 'diff_html', 'changes' list, and 'next_id'.
    """
    raw_words = raw_para.split()
    edited_words = edited_para.split()
    matcher = difflib.SequenceMatcher(None, raw_words, edited_words)

    segments = []
    cid = start_id
    context_words_count = 8

    opcodes = matcher.get_opcodes()
    pending_context = []

    for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        context_before = ""
        context_after = ""
        for prev_idx in range(op_idx - 1, -1, -1):
            ptag, _, _, pj1, pj2 = opcodes[prev_idx]
            if ptag in ("equal", "insert", "replace") and pj2 > pj1:
                words = edited_words[pj1:pj2]
                context_before = " ".join(words[-5:])
                break
        for next_idx in range(op_idx + 1, len(opcodes)):
            ntag, _, _, nj1, nj2 = opcodes[next_idx]
            if ntag in ("equal", "insert", "replace") and nj2 > nj1:
                words = edited_words[nj1:nj2]
                context_after = " ".join(words[:5])
                break

        if tag == "equal":
            equal_text = html_mod.escape(" ".join(raw_words[i1:i2]))
            pending_context.append(equal_text)
        else:
            seg_html_parts = []

            if pending_context:
                full_context = " ".join(pending_context)
                context_words_list = full_context.split()
                if len(context_words_list) > context_words_count:
                    earlier = " ".join(context_words_list[:-context_words_count])
                    segments.append({"html": earlier, "change": None})
                    seg_html_parts.append(
                        '<span style="color: #888;">…</span> '
                        + " ".join(context_words_list[-context_words_count:])
                    )
                else:
                    seg_html_parts.append(full_context)
                pending_context = []

            if tag == "delete":
                old_text = " ".join(raw_words[i1:i2])
                seg_html_parts.append(
                    f'<span style="background-color: #5c2020; text-decoration: line-through; padding: 1px 3px; border-radius: 3px;">'
                    f"{html_mod.escape(old_text)}</span>"
                )
                change_data = {
                    "id": cid,
                    "old": old_text,
                    "new": "",
                    "tag": "delete",
                    "context_before": context_before,
                    "context_after": context_after,
                }
                cid += 1
            elif tag == "insert":
                new_text = " ".join(edited_words[j1:j2])
                seg_html_parts.append(
                    f'<span style="background-color: #1e4620; font-weight: bold; padding: 1px 3px; border-radius: 3px;">'
                    f"{html_mod.escape(new_text)}</span>"
                )
                change_data = {
                    "id": cid,
                    "old": "",
                    "new": new_text,
                    "tag": "insert",
                    "context_before": context_before,
                    "context_after": context_after,
                }
                cid += 1
            elif tag == "replace":
                old_text = " ".join(raw_words[i1:i2])
                new_text = " ".join(edited_words[j1:j2])
                seg_html_parts.append(
                    f'<span style="background-color: #5c2020; text-decoration: line-through; padding: 1px 3px; border-radius: 3px;">'
                    f"{html_mod.escape(old_text)}</span> "
                    f'<span style="background-color: #1e4620; font-weight: bold; padding: 1px 3px; border-radius: 3px;">'
                    f"{html_mod.escape(new_text)}</span>"
                )
                change_data = {
                    "id": cid,
                    "old": old_text,
                    "new": new_text,
                    "tag": "replace",
                    "context_before": context_before,
                    "context_after": context_after,
                }
                cid += 1

            if op_idx + 1 < len(opcodes) and opcodes[op_idx + 1][0] == "equal":
                _, ni1, ni2, _, _ = opcodes[op_idx + 1]
                next_equal_words = raw_words[ni1:ni2]
                if len(next_equal_words) > context_words_count:
                    seg_html_parts.append(
                        " ".join(
                            html_mod.escape(w)
                            for w in next_equal_words[:context_words_count]
                        )
                    )
                    seg_html_parts.append(' <span style="color: #888;">…</span>')

            segments.append({"html": " ".join(seg_html_parts), "change": change_data})

    if pending_context:
        segments.append({"html": " ".join(pending_context), "change": None})

    all_changes = [s["change"] for s in segments if s["change"] is not None]

    return {"segments": segments, "changes": all_changes, "next_id": cid}


def apply_rejection(edited_text: str, change: dict) -> str:
    """
    Apply a single rejection: revert the edited text by undoing one change.
    - For 'delete': re-insert the old text using surrounding context as anchors
    - For 'insert': remove the new text
    - For 'replace': swap new text back to old text
    """
    if change["tag"] == "insert":
        return edited_text.replace(change["new"], "", 1)
    elif change["tag"] == "replace":
        return edited_text.replace(change["new"], change["old"], 1)
    elif change["tag"] == "delete":
        ctx_before = change.get("context_before", "")
        ctx_after = change.get("context_after", "")
        old_text = change["old"]

        if ctx_before and ctx_after:
            anchor = ctx_before + " " + ctx_after
            if anchor in edited_text:
                return edited_text.replace(
                    anchor, ctx_before + " " + old_text + " " + ctx_after, 1
                )

        if ctx_before and ctx_before in edited_text:
            idx = edited_text.index(ctx_before) + len(ctx_before)
            return edited_text[:idx] + " " + old_text + edited_text[idx:]

        if ctx_after and ctx_after in edited_text:
            idx = edited_text.index(ctx_after)
            return edited_text[:idx] + old_text + " " + edited_text[idx:]

        return edited_text + " " + old_text
    return edited_text
