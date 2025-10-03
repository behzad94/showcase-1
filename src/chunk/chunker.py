# src/chunk/chunker.py
# goal: split Doc.text into token-based windows (500 tokens, 50 overlap)
# we keep exact character slices so citations are honest and reproducible

from typing import List, Optional
from dataclasses import dataclass
import re
from src.ingest.ingest import Doc

@dataclass
class Chunk:
    chunk_id: str                 # unique key like "pdf::file.pdf::chunk0"
    doc_id: str                   # parent doc id (links back to Doc)
    filename: str                 # filename for citation display
    text: str                     # exact substring from original text
    token_count: int              # how many tokens in this chunk
    start_char: int               # char start offset in full text
    end_char: int                 # char end offset (non-inclusive)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

# token = any non-space sequence; this is simple but works ok for RAG windows
_TOKEN_RE = re.compile(r"\S+")

def token_spans(text: str) -> List[range]:
    # we return character ranges for each token so we can slice the text later
    spans: List[range] = []
    for m in _TOKEN_RE.finditer(text):
        spans.append(range(m.start(), m.end()))
    return spans

def _char_offset_to_page(cum_lengths: list[int], offset: int) -> int:
    # map a character offset to a page number using cumulative lengths
    if not cum_lengths:
        return 1
    for idx, upto in enumerate(cum_lengths, start=1):
        if offset < upto:
            return idx
    return len(cum_lengths)

def chunk_doc(doc: Doc, chunk_tokens: int = 500, overlap_tokens: int = 50) -> List[Chunk]:
    text = doc.text
    spans = token_spans(text)
    out: List[Chunk] = []
    if not spans:
        return out
    start_tok = 0
    chunk_idx = 0
    while start_tok < len(spans):
        end_tok = min(start_tok + chunk_tokens, len(spans))
        start_char = spans[start_tok].start
        end_char = spans[end_tok - 1].stop
        piece = text[start_char:end_char]
        tok_count = end_tok - start_tok

        cum = getattr(doc, "_cum_page_lengths", [])
        p_start = _char_offset_to_page(cum, start_char)
        p_end   = _char_offset_to_page(cum, end_char - 1)

        out.append(Chunk(
            chunk_id=f"{doc.id}::chunk{chunk_idx}",
            doc_id=doc.id,
            filename=doc.filename,
            text=piece,
            token_count=tok_count,
            start_char=start_char,
            end_char=end_char,
            page_start=p_start,
            page_end=p_end,
        ))
        chunk_idx += 1
        if end_tok == len(spans):
            break
        # move window with overlap
        start_tok = max(end_tok - overlap_tokens, start_tok + 1)
    return out

def chunk_corpus(docs: List[Doc], chunk_tokens: int = 500, overlap_tokens: int = 50) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_doc(d, chunk_tokens, overlap_tokens))
    return all_chunks

