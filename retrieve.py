"""Local semantic search over the project's saved chunk embeddings."""

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence

from embed_chunks import MODEL_DIMENSION, MODEL_NAME


PROJECT_DIR = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_DIR / "processed" / "embeddings.json"
REQUIRED_ITEM_FIELDS = ("chunk_id", "source", "page", "text", "embedding")
DEFAULT_SEMANTIC_WEIGHT = 0.75
DEFAULT_KEYWORD_WEIGHT = 0.25

# These are narrow, common corrections with an unambiguous intended form.
SAFE_CORRECTIONS = {
    "buisness": "business",
    "buisnesses": "businesses",
    "shld": "should",
}
STOPWORDS = {"a", "an", "are", "do", "for", "how", "i", "is", "me", "my", "the", "to", "what", "with", "you", "your", "want", "should"}
# A small query vocabulary bridge: achaar is the Indian term used in the source
# document for pickles. It only affects matching, never the stored passage text.
KEYWORD_SYNONYMS = {
    "pickle": {"pickle", "pickles", "achaar"},
    "pickles": {"pickle", "pickles", "achaar"},
    "achaar": {"pickle", "pickles", "achaar"},
}


def normalize_query(query: str) -> str:
    """Collapse whitespace and apply only conservative, explicit typo fixes."""
    normalized = " ".join(query.split())
    normalized = re.sub(r"\bwan(?=\s+to\b)", "want", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(a)(pickle)\b", r"\1 \2", normalized, flags=re.IGNORECASE)
    for misspelling, correction in SAFE_CORRECTIONS.items():
        normalized = re.sub(
            rf"\b{re.escape(misspelling)}\b",
            correction,
            normalized,
            flags=re.IGNORECASE,
        )
    return normalized


def keyword_match_score(query: str, passage: str) -> float:
    """Return query-term coverage, counting the small synonym set as matches."""
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in STOPWORDS
    }
    if not query_tokens:
        return 0.0
    passage_tokens = set(re.findall(r"[a-z0-9]+", passage.lower()))
    matched = 0
    for token in query_tokens:
        equivalents = KEYWORD_SYNONYMS.get(token, {token})
        if equivalents.intersection(passage_tokens):
            matched += 1
    return matched / len(query_tokens)


def _validated_vector(vector: Sequence[float], dimension: int, label: str) -> list[float]:
    """Convert a vector to floats and reject malformed or zero vectors."""
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only numeric values") from exc
    if len(values) != dimension:
        raise ValueError(f"{label} has dimension {len(values)}; expected {dimension}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must contain only finite values")
    if math.sqrt(math.fsum(value * value for value in values)) == 0:
        raise ValueError(f"{label} cannot be a zero vector")
    return values


def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    """Load and validate the JSON embedding index without loading chunks.json."""
    if not path.is_file():
        raise FileNotFoundError(f"Embedding index not found: {path}")
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse embedding index {path}: {exc}") from exc

    if not isinstance(index, dict):
        raise ValueError("Embedding index must be a JSON object")
    if index.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Index model is {index.get('model_name')!r}, expected {MODEL_NAME!r}"
        )
    dimension = index.get("embedding_dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError("Embedding index has an invalid embedding_dimension")
    if index.get("normalized") is not True:
        raise ValueError("Embedding index must declare normalized vectors")
    items = index.get("items")
    if not isinstance(items, list):
        raise ValueError("Embedding index 'items' must be a list")
    if index.get("count") != len(items):
        raise ValueError("Embedding index count does not match the number of items")

    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Index item {position} must be a JSON object")
        missing = [field for field in REQUIRED_ITEM_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Index item {position} is missing fields: {', '.join(missing)}")
        if not isinstance(item["text"], str):
            raise ValueError(f"Index item {position} text must be a string")
        vector = _validated_vector(item["embedding"], dimension, f"Index item {position} vector")
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Index item {position} vector is not L2-normalized")
    return index


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two equal-length, finite, nonzero vectors."""
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} and {len(right)}")
    left_values = _validated_vector(left, len(left), "Query vector")
    right_values = _validated_vector(right, len(left), "Document vector")
    dot = math.fsum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(math.fsum(value * value for value in left_values))
    right_norm = math.sqrt(math.fsum(value * value for value in right_values))
    return dot / (left_norm * right_norm)


def load_query_encoder() -> Callable[[str], Sequence[float]]:
    """Load the existing sentence-transformer model from disk without network access."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is missing. Install requirements with "
            "'py -m pip install -r requirements.txt'."
        ) from exc

    try:
        model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Model {MODEL_NAME} is not fully available in the local cache. "
            "Run the embedding stage once with internet access before offline retrieval."
        ) from exc
    dimension = model.get_embedding_dimension()
    if dimension != MODEL_DIMENSION:
        raise RuntimeError(
            f"Model dimension mismatch: expected {MODEL_DIMENSION}, got {dimension}"
        )

    def encode_query(text: str) -> Sequence[float]:
        return model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    return encode_query


def search(
    query: str,
    index: dict[str, Any],
    encode_query: Callable[[str], Sequence[float]],
    top_k: int = 5,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[dict[str, Any]]:
    """Rank passages by a weighted semantic and lexical relevance score.

    ``score`` remains an alias for ``semantic_score`` for compatibility.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not all(math.isfinite(weight) and weight >= 0 for weight in (semantic_weight, keyword_weight)):
        raise ValueError("Ranking weights must be finite, non-negative numbers")
    if not math.isclose(semantic_weight + keyword_weight, 1.0, abs_tol=1e-9):
        raise ValueError("semantic_weight and keyword_weight must sum to 1.0")
    items = index.get("items")
    if not isinstance(items, list):
        raise ValueError("Embedding index 'items' must be a list")
    if not items:
        return []

    dimension = index.get("embedding_dimension")
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding index has an invalid embedding_dimension")
    normalized_query = normalize_query(query)
    query_vector = _validated_vector(encode_query(normalized_query), dimension, "Query vector")
    scored = []
    for position, item in enumerate(items, start=1):
        vector = _validated_vector(item.get("embedding", []), dimension, f"Index item {position} vector")
        semantic_score = cosine_similarity(query_vector, vector)
        keyword_score = keyword_match_score(normalized_query, item["text"])
        ranking_score = semantic_weight * semantic_score + keyword_weight * keyword_score
        scored.append(
            {
                "score": semantic_score,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "ranking_score": ranking_score,
                "chunk_id": item["chunk_id"],
                "source": item["source"],
                "page": item["page"],
                "text": item["text"],
            }
        )
    scored.sort(key=lambda result: result["ranking_score"], reverse=True)
    return scored[: min(top_k, len(scored))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Malkin AI's local PDF chunks.")
    parser.add_argument("question", nargs="*", help="Natural-language search question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=DEFAULT_SEMANTIC_WEIGHT,
        help=f"Semantic score weight (default: {DEFAULT_SEMANTIC_WEIGHT})",
    )
    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=DEFAULT_KEYWORD_WEIGHT,
        help=f"Keyword coverage weight (default: {DEFAULT_KEYWORD_WEIGHT})",
    )
    args = parser.parse_args()
    question = " ".join(args.question).strip() or input("Ask a question: ").strip()

    try:
        index = load_index()
        if not index["items"]:
            print("The embedding index is empty.")
            return
        encoder = load_query_encoder()
        results = search(
            question,
            index,
            encoder,
            args.top_k,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
        )
    except Exception as exc:
        parser.exit(1, f"Retrieval error: {exc}\n")

    if not results:
        print("No matching passages found.")
        return
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. ranking={result['ranking_score']:.4f} | "
            f"semantic={result['semantic_score']:.4f} | "
            f"keyword={result['keyword_score']:.4f} | "
            f"{result['source']} | page {result['page']} | {result['chunk_id']}"
        )
        print(result["text"].strip())


if __name__ == "__main__":
    main()
