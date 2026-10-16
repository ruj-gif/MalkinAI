"""Split extracted PDF page text into overlapping, page-scoped chunks."""

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "processed" / "extracted_pages.json"
OUTPUT_PATH = PROJECT_DIR / "processed" / "chunks.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
PREVIEW_LENGTH = 180


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size windows with a character overlap.

    Short or whitespace-only text produces one chunk or no chunks respectively.
    The final chunk may be shorter than ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size")
    if not text or not text.strip():
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def create_chunks(page_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create uniquely identified chunks while keeping each page independent."""
    result = []
    for record in page_records:
        source = record.get("source")
        page = record.get("page")
        text = record.get("text", "")
        if not isinstance(source, str) or not isinstance(page, int):
            raise ValueError("Each page record must include a string source and integer page")
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string for {source}, page {page}")

        for page_chunk_number, chunk in enumerate(split_text(text), start=1):
            result.append(
                {
                    "source": source,
                    "page": page,
                    "chunk_id": f"chunk-{len(result) + 1:06d}",
                    "text": chunk,
                }
            )
    return result


def main() -> None:
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Extracted page JSON not found: {INPUT_PATH}")

    try:
        page_records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {INPUT_PATH}: {exc}") from exc
    if not isinstance(page_records, list):
        raise SystemExit(f"Expected a JSON list of page records in {INPUT_PATH}")

    try:
        chunks = create_chunks(page_records)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid page records in {INPUT_PATH}: {exc}") from exc

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(chunk["source"] for chunk in chunks)
    print(f"Page records loaded: {len(page_records)}")
    print(f"Chunks created: {len(chunks)}")
    print("Chunks per source PDF:")
    for source in sorted({record.get("source", "(unknown)") for record in page_records}):
        print(f"  {source}: {counts.get(source, 0)}")

    print("Chunk previews:")
    for chunk in chunks[:3]:
        preview = " ".join(chunk["text"].split())
        if len(preview) > PREVIEW_LENGTH:
            preview = preview[:PREVIEW_LENGTH].rstrip() + "..."
        print(f"  {chunk['chunk_id']} | {chunk['source']} | page {chunk['page']}: {preview}")
    print(f"Saved chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
