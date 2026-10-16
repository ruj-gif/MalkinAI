"""Generate normalized sentence embeddings for the project's text chunks."""

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "processed" / "chunks.json"
OUTPUT_PATH = PROJECT_DIR / "processed" / "embeddings.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
BATCH_SIZE = 32


def non_empty_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep valid chunk records whose text contains non-whitespace characters."""
    return [
        chunk
        for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]


def build_output(
    chunks: list[dict[str, Any]],
    vectors: Sequence[Sequence[float]],
    model_name: str = MODEL_NAME,
    expected_dimension: int | None = MODEL_DIMENSION,
) -> dict[str, Any]:
    """Validate vectors and combine them with source chunk metadata."""
    selected = non_empty_chunks(chunks)
    if len(selected) != len(vectors):
        raise ValueError(f"Expected {len(selected)} vectors, received {len(vectors)}")

    if not selected:
        dimension = expected_dimension
    else:
        dimension = len(vectors[0])
        if dimension == 0:
            raise ValueError("Embedding vectors cannot be empty")
        if expected_dimension is not None and dimension != expected_dimension:
            raise ValueError(
                f"Expected vectors of dimension {expected_dimension}, received {dimension}"
            )

    items = []
    for chunk, vector in zip(selected, vectors):
        if len(vector) != dimension:
            raise ValueError("All embedding vectors must have the same dimension")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding vectors must contain only finite numbers")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError("Embedding vectors must be L2-normalized")
        for key in ("chunk_id", "source", "page", "text"):
            if key not in chunk:
                raise ValueError(f"Chunk record is missing required field: {key}")
        items.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "text": chunk["text"],
                "embedding": values,
            }
        )

    return {
        "model_name": model_name,
        "embedding_dimension": dimension,
        "normalized": True,
        "similarity_metric": "cosine",
        "count": len(items),
        "items": items,
    }


def generate_embeddings(
    chunks: list[dict[str, Any]],
    encode: Callable[[list[str]], Sequence[Sequence[float]]],
) -> dict[str, Any]:
    """Encode non-empty chunks with the supplied encoder and validate results."""
    selected = non_empty_chunks(chunks)
    vectors = encode([chunk["text"] for chunk in selected]) if selected else []
    return build_output(selected, vectors)


def load_model_encoder() -> Callable[[list[str]], Sequence[Sequence[float]]]:
    """Load the model on CPU, downloading it from Hugging Face if not cached."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is missing. Install requirements with "
            "'py -m pip install -r requirements.txt'."
        ) from exc

    print(f"Loading {MODEL_NAME} on CPU (first run downloads the model if needed)...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    actual_dimension = model.get_embedding_dimension()
    if actual_dimension != MODEL_DIMENSION:
        raise RuntimeError(
            f"Model dimension changed: expected {MODEL_DIMENSION}, got {actual_dimension}"
        )

    def encode(texts: list[str]) -> Sequence[Sequence[float]]:
        return model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

    return encode


def main() -> None:
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Chunk JSON not found: {INPUT_PATH}")
    try:
        chunks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {INPUT_PATH}: {exc}") from exc
    if not isinstance(chunks, list):
        raise SystemExit(f"Expected a JSON list of chunks in {INPUT_PATH}")

    selected = non_empty_chunks(chunks)
    if selected:
        try:
            output = generate_embeddings(selected, load_model_encoder())
        except Exception as exc:
            raise SystemExit(
                f"Embedding generation failed. Check the installation and internet access "
                f"for the first model download. Details: {exc}"
            ) from exc
    else:
        output = build_output([], [])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Non-empty chunks embedded: {output['count']}")
    print(f"Model: {output['model_name']}")
    print(f"Embedding dimension: {output['embedding_dimension']}")
    print("Vectors L2-normalized: yes (cosine similarity)")
    print(f"Saved embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
