"""Split documents into overlapping chunks for retrieval."""


def chunk_text(text, chunk_size=500, overlap=75):
    """
    Sliding-window chunker over characters.

    Tries to cut on sentence boundaries so chunks stay readable.
    Returns a list of chunk strings.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # prefer to cut at the last sentence end inside the window
            cut = max(text.rfind(". ", start, end),
                      text.rfind(".\n", start, end),
                      text.rfind("\n\n", start, end))
            if cut > start + chunk_size // 3:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap
    return chunks
