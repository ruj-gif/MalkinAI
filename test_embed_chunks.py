"""Offline-friendly tests for embedding output validation and metadata."""

import unittest

from embed_chunks import build_output, generate_embeddings


def unit_vector(*values):
    return list(values)


class EmbeddingOutputTests(unittest.TestCase):
    def test_output_shape_and_metadata_are_preserved(self):
        chunk = {
            "chunk_id": "chunk-1",
            "source": "scheme.pdf",
            "page": 4,
            "text": "A useful chunk",
        }
        output = build_output([chunk], [unit_vector(0.6, 0.8)], expected_dimension=2)
        self.assertEqual(output["embedding_dimension"], 2)
        self.assertEqual(output["count"], 1)
        self.assertEqual(
            {key: output["items"][0][key] for key in ("chunk_id", "source", "page", "text")},
            chunk,
        )
        self.assertEqual(output["items"][0]["embedding"], [0.6, 0.8])

    def test_empty_input_is_valid(self):
        output = build_output([], [])
        self.assertEqual(output["items"], [])
        self.assertEqual(output["count"], 0)
        self.assertEqual(output["embedding_dimension"], 384)

    def test_generate_skips_empty_chunks_and_calls_encoder(self):
        chunks = [
            {"chunk_id": "empty", "source": "a.pdf", "page": 1, "text": "  "},
            {"chunk_id": "kept", "source": "a.pdf", "page": 2, "text": "words"},
        ]
        seen = []

        def encoder(texts):
            seen.extend(texts)
            return [unit_vector(1.0, *([0.0] * 383))]

        output = generate_embeddings(chunks, encoder)
        self.assertEqual(seen, ["words"])
        self.assertEqual([item["chunk_id"] for item in output["items"]], ["kept"])

    def test_rejects_inconsistent_or_non_normalized_vectors(self):
        chunks = [
            {"chunk_id": "one", "source": "a.pdf", "page": 1, "text": "first"},
            {"chunk_id": "two", "source": "a.pdf", "page": 1, "text": "second"},
        ]
        with self.assertRaises(ValueError):
            build_output(chunks, [[1.0, 0.0], [0.0, 1.0, 0.0]], expected_dimension=None)
        with self.assertRaises(ValueError):
            build_output(chunks[:1], [[2.0, 0.0]], expected_dimension=2)


if __name__ == "__main__":
    unittest.main()
