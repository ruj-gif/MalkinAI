"""Unit tests for the page text chunking stage."""

import unittest

from chunk_text import create_chunks, split_text


class SplitTextTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(split_text("A short page."), ["A short page."])

    def test_empty_and_whitespace_text_make_no_chunks(self):
        self.assertEqual(split_text(""), [])
        self.assertEqual(split_text(" \n\t "), [])

    def test_long_text_is_split_with_requested_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = split_text(text, chunk_size=10, overlap=3)
        self.assertEqual(chunks, ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"])
        for previous, following in zip(chunks, chunks[1:]):
            self.assertEqual(previous[-3:], following[:3])

    def test_bad_chunk_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=0)
        with self.assertRaises(ValueError):
            split_text("text", chunk_size=5, overlap=5)


class MetadataTests(unittest.TestCase):
    def test_metadata_is_preserved_and_chunk_ids_are_unique(self):
        records = [
            {"source": "scheme.pdf", "page": 7, "text": "x" * 20},
            {"source": "scheme.pdf", "page": 8, "text": "another page"},
        ]
        chunks = create_chunks(records)
        self.assertEqual(chunks[0]["source"], "scheme.pdf")
        self.assertEqual(chunks[0]["page"], 7)
        self.assertEqual(chunks[-1]["page"], 8)
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))
        self.assertTrue(all(set(chunk) == {"source", "page", "chunk_id", "text"} for chunk in chunks))

    def test_pages_are_not_merged(self):
        chunks = create_chunks(
            [
                {"source": "a.pdf", "page": 1, "text": "first"},
                {"source": "b.pdf", "page": 2, "text": "second"},
            ]
        )
        self.assertEqual([(c["source"], c["page"]) for c in chunks], [("a.pdf", 1), ("b.pdf", 2)])


if __name__ == "__main__":
    unittest.main()
