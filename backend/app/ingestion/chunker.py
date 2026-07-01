"""Character-window chunking with overlap.

Kept deliberately simple and deterministic so that chunk size / overlap are
clean experiment knobs. We split on paragraph boundaries first and pack
paragraphs into windows, which keeps table rows together better than a blind
character slice.
"""


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf = f"{buf}\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            # Start the next buffer with a tail overlap of the previous one.
            tail = buf[-overlap:] if overlap and buf else ""
            buf = f"{tail}\n{para}".strip() if tail else para
            # A single very long paragraph still needs hard splitting.
            while len(buf) > chunk_size:
                chunks.append(buf[:chunk_size])
                buf = buf[chunk_size - overlap :]

    if buf:
        chunks.append(buf)
    return chunks
