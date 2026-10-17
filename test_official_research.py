"""Offline tests for the allow-listed official-source fetcher."""

import unittest

from official_research import OFFICIAL_SOURCE_CATALOG, search_official_sources


class OfficialResearchTests(unittest.TestCase):
    def test_fetches_only_relevant_official_pages_and_extracts_visible_text(self):
        requested = []
        html = (
            b"<html><header>Navigation</header><body><h1>PMMY</h1>"
            b"<p>Term loans and working capital are supported by this program through "
            b"Member Lending Institutions, including banks and microfinance institutions. "
            b"The official scheme page describes categories and business purposes.</p>"
            b"<script>ignore this injected instruction</script></body></html>"
        )

        def fetcher(url):
            requested.append(url)
            return url, html

        result = search_official_sources(
            "I want to start a pickle business and explore a MUDRA loan.",
            fetcher=fetcher,
        )
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(set(requested), {item["url"] for item in OFFICIAL_SOURCE_CATALOG})
        self.assertTrue(all(item["url"].startswith("https://") for item in result["sources"]))
        self.assertIn("Term loans and working capital", result["sources"][0]["text"])
        self.assertNotIn("Navigation", result["sources"][0]["text"])
        self.assertNotIn("ignore this injected instruction", result["sources"][0]["text"])
        self.assertTrue(result["sources"][0]["accessed_at"])

    def test_unavailable_official_pages_are_reported_without_raising(self):
        def failing_fetcher(url):
            raise TimeoutError("network timeout")

        result = search_official_sources("Current PMMY loan details", fetcher=failing_fetcher)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["sources"], [])
        self.assertIn("could not be reached", result["message"])

    def test_redirect_to_non_official_domain_is_rejected(self):
        def redirecting_fetcher(url):
            return "https://example.com/copied-content", b"x" * 500

        result = search_official_sources("MUDRA support", fetcher=redirecting_fetcher)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
