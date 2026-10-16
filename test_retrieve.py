"""Offline tests for semantic retrieval using small fake vectors."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from embed_chunks import MODEL_DIMENSION, MODEL_NAME
from retrieve import (
    PROJECT_DIR,
    cosine_similarity,
    keyword_match_score,
    load_index,
    load_query_encoder,
    normalize_query,
    search,
)


def make_index(items, dimension=2):
    return {
        "model_name": MODEL_NAME,
        "embedding_dimension": dimension,
        "normalized": True,
        "similarity_metric": "cosine",
        "count": len(items),
        "items": items,
    }


def item(chunk_id, source, page, text, vector):
    return {
        "chunk_id": chunk_id,
        "source": source,
        "page": page,
        "text": text,
        "embedding": vector,
    }


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = make_index(
            [
                item("c1", "scheme-a.pdf", 3, "Loan support for new enterprises.", [1.0, 0.0]),
                item("c2", "scheme-b.pdf", 8, "Women entrepreneur subsidy.", [0.0, 1.0]),
                item("c3", "scheme-c.pdf", 2, "Business loan and credit support.", [0.8, 0.6]),
            ]
        )

    def test_ranking_and_metadata(self):
        results = search("loan question", self.index, lambda _: [1.0, 0.0], top_k=2)
        self.assertEqual([result["chunk_id"] for result in results], ["c1", "c3"])
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])
        self.assertEqual(
            {key: results[0][key] for key in ("source", "page", "text")},
            {"source": "scheme-a.pdf", "page": 3, "text": "Loan support for new enterprises."},
        )
        self.assertIn("score", results[0])

    def test_query_normalization_only_applies_safe_corrections(self):
        self.assertEqual(
            normalize_query("  i   wan to start apickle buisness what shld i do "),
            "i want to start a pickle business what should i do",
        )
        self.assertEqual(normalize_query("small business support"), "small business support")

    def test_keyword_matches_achaar_as_pickle_synonym(self):
        self.assertAlmostEqual(
            keyword_match_score("start pickle business", "papad making, achaar making"),
            1 / 3,
        )

    def test_mixed_ranking_uses_both_semantic_and_keyword_scores(self):
        mixed_index = make_index(
            [
                item("semantic", "general.pdf", 1, "Seasonal orchard irrigation guidance.", [0.6, 0.8]),
                item("keyword", "pickle.pdf", 4, "Papad making and achaar making.", [0.55, (1 - 0.55**2) ** 0.5]),
            ]
        )
        results = search(
            "i wan to start apickle buisness what shld i do",
            mixed_index,
            lambda _: [1.0, 0.0],
            top_k=2,
        )
        self.assertEqual(results[0]["chunk_id"], "keyword")
        self.assertGreater(results[0]["keyword_score"], results[1]["keyword_score"])
        self.assertEqual(results[0]["semantic_score"], results[0]["score"])
        self.assertAlmostEqual(
            results[0]["ranking_score"],
            0.75 * results[0]["semantic_score"] + 0.25 * results[0]["keyword_score"],
        )

    def test_actual_achaar_chunk_ranks_for_pickle_business_query(self):
        chunks_path = Path(__file__).resolve().parent / "processed" / "chunks.json"
        stored_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        passage = next(chunk for chunk in stored_chunks if chunk["chunk_id"] == "chunk-000176")
        self.assertIn("achaar making", passage["text"].lower())
        test_index = make_index(
            [
                item("general", "general.pdf", 1, "Seasonal orchard irrigation guidance.", [0.6, 0.8]),
                item(
                    passage["chunk_id"],
                    passage["source"],
                    passage["page"],
                    passage["text"],
                    [0.55, (1 - 0.55**2) ** 0.5],
                ),
            ]
        )
        results = search(
            "i wan to start apickle buisness what shld i do",
            test_index,
            lambda _: [1.0, 0.0],
            top_k=2,
        )
        self.assertEqual(results[0]["chunk_id"], "chunk-000176")
        self.assertEqual(results[0]["source"], "Mudra_Offerings.pdf")
        self.assertEqual(results[0]["page"], 4)

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimension"):
            search("query", self.index, lambda _: [1.0])
        bad_index = make_index([item("bad", "a.pdf", 1, "bad", [1.0, 0.0, 0.0])])
        with self.assertRaisesRegex(ValueError, "dimension"):
            search("query", bad_index, lambda _: [1.0, 0.0])

    def test_empty_query_and_top_k_validation(self):
        with self.assertRaisesRegex(ValueError, "Query"):
            search("  ", self.index, lambda _: [1.0, 0.0])
        for value in (0, -1, 1.5, True):
            with self.subTest(top_k=value), self.assertRaisesRegex(ValueError, "top_k"):
                search("query", self.index, lambda _: [1.0, 0.0], top_k=value)

    def test_metadata_and_text_are_not_modified_by_normalization(self):
        original = item("c1", "exact source.pdf", 9, "achaar making   ", [1.0, 0.0])
        results = search("pickle business", make_index([original]), lambda _: [1.0, 0.0])
        self.assertEqual(
            {key: results[0][key] for key in ("chunk_id", "source", "page", "text")},
            {key: original[key] for key in ("chunk_id", "source", "page", "text")},
        )

    def test_top_k_above_index_size_returns_available_results(self):
        results = search("query", self.index, lambda _: [1.0, 0.0], top_k=20)
        self.assertEqual(len(results), 3)

    def test_empty_index_returns_empty_without_encoding(self):
        results = search("valid question", make_index([]), lambda _: self.fail("encoded"))
        self.assertEqual(results, [])

    def test_cosine_similarity_handles_non_normalized_inputs(self):
        self.assertAlmostEqual(cosine_similarity([3.0, 0.0], [1.0, 1.0]), 2 ** -0.5)

    def test_load_index_reads_the_structured_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / "processed") as temp_dir:
            path = Path(temp_dir) / "index.json"
            path.write_text(json.dumps(self.index), encoding="utf-8")
            self.assertEqual(load_index(path)["count"], 3)

    def test_checked_in_project_index_loads_and_returns_search_results(self):
        index = load_index(PROJECT_DIR / "processed" / "embeddings.json")
        self.assertEqual(index["count"], 212)
        results = search(
            "business support",
            index,
            lambda _: [1.0] + [0.0] * (MODEL_DIMENSION - 1),
            top_k=2,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["source"] and result["page"] >= 1 for result in results))

    def test_query_encoder_downloads_model_if_deployment_cache_is_empty(self):
        calls = []

        class FakeVector:
            def tolist(self):
                return [1.0] + [0.0] * (MODEL_DIMENSION - 1)

        class FakeModel:
            def get_embedding_dimension(self):
                return MODEL_DIMENSION

            def encode(self, *args, **kwargs):
                return [FakeVector()]

        def fake_sentence_transformer(model_name, device, **kwargs):
            calls.append((model_name, device, kwargs))
            if kwargs.get("local_files_only"):
                raise OSError("model is not cached")
            return FakeModel()

        module = SimpleNamespace(SentenceTransformer=fake_sentence_transformer)
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            encoder = load_query_encoder()

        self.assertEqual(encoder("support"), [1.0] + [0.0] * (MODEL_DIMENSION - 1))
        self.assertTrue(calls[0][2]["local_files_only"])
        self.assertEqual(calls[1][0], MODEL_NAME)
        self.assertEqual(calls[1][1], "cpu")
        self.assertNotIn("local_files_only", calls[1][2])

    def test_load_index_rejects_invalid_vector_and_missing_file(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent / "processed") as temp_dir:
            path = Path(temp_dir) / "index.json"
            bad_index = make_index([item("bad", "a.pdf", 1, "bad", [0.0, 0.0])])
            path.write_text(json.dumps(bad_index), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero vector|normalized"):
                load_index(path)
            with self.assertRaises(FileNotFoundError):
                load_index(Path(temp_dir) / "missing.json")


if __name__ == "__main__":
    unittest.main()
