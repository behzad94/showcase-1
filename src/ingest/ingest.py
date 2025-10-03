# src/ingest/ingest.py
# goal: read files from "corpus/" and return Doc objects with clean text
# note: supports PDF/TXT/MD; for PDF we also keep page numbers length to map pages later

from dataclasses import dataclass                 # simple data container
from typing import List, Optional                 # type hints for clarity
from pathlib import Path                          # file paths (portable)
from datetime import datetime                     # ISO timestamps for metadata
import chardet                                    # guess encoding for TXT
import fitz                                       # PyMuPDF for PDF reading

@dataclass
class Doc:
    id: str                       # unique id, like "pdf::filename.pdf"
    filename: str                 # plain filename for citations
    filetype: str                 # 'pdf' | 'txt' | 'md'
    text: str                     # full plain text content
    pages: Optional[List[int]]    # page numbers (for PDF), else None
    created_at: str               # ISO timestamp when we created this record

def now_iso() -> str:
    # make a consistent ISO timestamp (useful in logs/debug)
    return datetime.now().isoformat()

def read_pdf(pdf_path: Path) -> Doc:
    # make sure extension is right (early safety)
    assert pdf_path.suffix.lower() == ".pdf", "file must be .pdf"
    page_texts: List[str] = []       # collect text per page
    page_numbers: List[int] = []     # keep page indices 1..N
    with fitz.open(pdf_path) as doc: # open PDF
        for i in range(doc.page_count):             # loop pages
            page = doc.load_page(i)                 # get page i
            text = page.get_text("text") or ""      # extract plain text
            page_texts.append(text)                 # keep text
            page_numbers.append(i + 1)              # human page number
    full_text = "\n\n".join(page_texts)             # merge pages with gaps

    # build cumulative char lengths so we can map char offsets → page numbers
    cum_lengths: List[int] = []
    total = 0
    for t in page_texts:
        total += len(t) + 2                          # +2 for the \n\n join
        cum_lengths.append(total)

    d = Doc(
        id=f"pdf::{pdf_path.name}",
        filename=pdf_path.name,
        filetype="pdf",
        text=full_text,
        pages=page_numbers,
        created_at=now_iso(),
    )
    # store page length helper on the instance (used by chunker)
    setattr(d, "_cum_page_lengths", cum_lengths)
    return d

def read_txt(txt_path: Path) -> Doc:
    assert txt_path.suffix.lower() == ".txt", "file must be .txt"
    raw = txt_path.read_bytes()                      # read raw bytes
    detected = chardet.detect(raw)                   # guess encoding
    encoding = detected.get("encoding") or "utf-8"   # default utf-8
    text = raw.decode(encoding, errors="replace")    # decode safely
    text = "\n".join(line.rstrip() for line in text.splitlines())  # trim right
    return Doc(
        id=f"txt::{txt_path.name}",
        filename=txt_path.name,
        filetype="txt",
        text=text,
        pages=None,
        created_at=now_iso(),
    )

def read_md(md_path: Path) -> Doc:
    assert md_path.suffix.lower() == ".md", "file must be .md"
    raw = md_path.read_bytes()
    encoding = (chardet.detect(raw).get("encoding") or "utf-8")
    text = raw.decode(encoding, errors="replace")

    # very light markdown cleanup (we want plain text for embedding)
    text = text.replace("#", "").replace("*", "").replace(">", "").replace("-", "")
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return Doc(
        id=f"md::{md_path.name}",
        filename=md_path.name,
        filetype="md",
        text=text,
        pages=None,
        created_at=now_iso(),
    )

def scan_corpus_dir(corpus_dir: Path) -> List[Doc]:
    # check folder exists to avoid surprises
    assert corpus_dir.exists(), f"corpus not found: {corpus_dir}"
    docs: List[Doc] = []
    for path in corpus_dir.iterdir():
        if not path.is_file():
            continue
        sfx = path.suffix.lower()
        if sfx == ".pdf":
            docs.append(read_pdf(path))
        elif sfx == ".txt":
            docs.append(read_txt(path))
        elif sfx == ".md":
            docs.append(read_md(path))
        else:
            print(f"[info] skip unsupported: {path.name}")
    print(f"[info] loaded {len(docs)} doc(s) from corpus")
    return docs

