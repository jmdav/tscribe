import re


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

    chunks_with_context = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            chunks_with_context.append(("", chunk))
        else:
            prev_chunk_words = raw_chunks[i - 1].split()
            context = " ".join(prev_chunk_words[-context_words:])
            chunks_with_context.append((context, chunk))

    return chunks_with_context
