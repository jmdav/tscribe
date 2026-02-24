import re
import difflib
import html as html_mod


def build_paragraph_diffs(raw_text: str, edited_text: str):
    """
    Build continuous diff data that preserves newlines as visible changes.
    Returns a list with a single dict containing:
      - 'segments': the text with inline diff highlighting
      - 'changes': list of individual changes [{id, old, new, tag}]
    """
    # Tokenize entire text while preserving newlines as separate tokens
    raw_tokens = re.findall(r"\S+|\n", raw_text)
    edited_tokens = re.findall(r"\S+|\n", edited_text)

    matcher = difflib.SequenceMatcher(None, raw_tokens, edited_tokens)

    segments = []
    change_id = 0
    opcodes = matcher.get_opcodes()

    def join_tokens(tokens):
        """Join tokens, using actual newlines for \n tokens and spaces for words."""
        result = []
        for i, token in enumerate(tokens):
            if token == "\n":
                result.append("\n")
            else:
                if i > 0 and tokens[i - 1] != "\n":
                    result.append(" ")
                result.append(token)
        return "".join(result)

    def escape_with_newlines(text):
        """Escape HTML and convert newlines to <br> tags."""
        escaped = html_mod.escape(text)
        return escaped.replace("\n", "<br>")

    # Pre-compute context for all opcodes in O(n) instead of O(n²)
    context_before_map = {}
    context_after_map = {}
    last_context = ""
    for idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        context_before_map[idx] = last_context
        if tag in ("equal", "insert", "replace") and j2 > j1:
            words = [w for w in edited_tokens[j1:j2] if w != "\n"]
            last_context = " ".join(words[-5:])
    # Reverse pass for context_after
    last_context = ""
    for idx in range(len(opcodes) - 1, -1, -1):
        tag, i1, i2, j1, j2 = opcodes[idx]
        context_after_map[idx] = last_context
        if tag in ("equal", "insert", "replace") and j2 > j1:
            words = [w for w in edited_tokens[j1:j2] if w != "\n"]
            last_context = " ".join(words[:5])

    for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        context_before = context_before_map[op_idx]
        context_after = context_after_map[op_idx]

        if tag == "equal":
            # Context text - not clickable
            equal_text = escape_with_newlines(join_tokens(raw_tokens[i1:i2]))
            segments.append({"html": equal_text, "change": None})
        elif tag == "delete":
            old_text = join_tokens(raw_tokens[i1:i2])
            change_html = (
                f'<span style="background-color: #5c2020; text-decoration: line-through; padding: 1px 3px; border-radius: 3px;">'
                f"{escape_with_newlines(old_text)}</span>"
            )
            change_data = {
                "id": change_id,
                "old": old_text,
                "new": "",
                "tag": "delete",
                "context_before": context_before,
                "context_after": context_after,
            }
            segments.append({"html": change_html, "change": change_data})
            change_id += 1
        elif tag == "insert":
            new_text = join_tokens(edited_tokens[j1:j2])
            change_html = (
                f'<span style="background-color: #1e4620; font-weight: bold; padding: 1px 3px; border-radius: 3px;">'
                f"{escape_with_newlines(new_text)}</span>"
            )
            change_data = {
                "id": change_id,
                "old": "",
                "new": new_text,
                "tag": "insert",
                "context_before": context_before,
                "context_after": context_after,
            }
            segments.append({"html": change_html, "change": change_data})
            change_id += 1
        elif tag == "replace":
            old_text = join_tokens(raw_tokens[i1:i2])
            new_text = join_tokens(edited_tokens[j1:j2])
            change_html = (
                f'<span style="background-color: #5c2020; text-decoration: line-through; padding: 1px 3px; border-radius: 3px;">'
                f"{escape_with_newlines(old_text)}</span> "
                f'<span style="background-color: #1e4620; font-weight: bold; padding: 1px 3px; border-radius: 3px;">'
                f"{escape_with_newlines(new_text)}</span>"
            )
            change_data = {
                "id": change_id,
                "old": old_text,
                "new": new_text,
                "tag": "replace",
                "context_before": context_before,
                "context_after": context_after,
            }
            segments.append({"html": change_html, "change": change_data})
            change_id += 1

    all_changes = [s["change"] for s in segments if s["change"] is not None]

    # Return as a single-paragraph result to maintain the same interface
    return [{"segments": segments, "changes": all_changes}]


def apply_rejection(edited_text: str, change: dict) -> str:
    """
    Apply a single rejection: revert the edited text by undoing one change.
    - For 'delete': re-insert the old text using surrounding context as anchors
    - For 'insert': remove the new text
    - For 'replace': swap new text back to old text
    """
    if isinstance(change, dict) and "tag" not in change:
        inner_change = change.get("change")
        if isinstance(inner_change, dict) and "tag" in inner_change:
            change = inner_change

    tag = change.get("tag") if isinstance(change, dict) else None
    if tag == "insert":
        new_text = change.get("new", "")
        return edited_text.replace(new_text, "", 1) if new_text else edited_text
    elif tag == "replace":
        new_text = change.get("new", "")
        old_text = change.get("old", "")
        if not new_text or not old_text:
            return edited_text
        return edited_text.replace(new_text, old_text, 1)
    elif tag == "delete":
        ctx_before = change.get("context_before", "")
        ctx_after = change.get("context_after", "")
        old_text = change.get("old", "")

        if not old_text:
            return edited_text

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
