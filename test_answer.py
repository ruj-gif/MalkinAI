"""Answer generation tests with mocked Groq chat-completions responses."""

import os
import json
import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from answer import (
    DEFAULT_MODEL_ID,
    ENV_PATH,
    AnswerError,
    SYSTEM_PROMPT,
    SUSPICIOUS_AMOUNT_PATTERN,
    _load_configuration,
    _contains_corrupt_amount,
    answer_question,
    search_official_sources,
)


SAMPLE_PASSAGE = {
    "source": "Sample Scheme.pdf",
    "page": 3,
    "chunk_id": "chunk-000003",
    "text": "A retrieved passage about a business support program.",
    "ranking_score": 0.8,
}


def response_json(documented=None, uncertain=None, suggestions=None, follow_up=None):
    payload = {
        "documented": documented or [{
            "claim": "A retrieved passage describes business support",
            "evidence_ids": ["chunk-000003"],
        }],
        "uncertain": uncertain or [],
        "suggestions": suggestions or [],
    }
    if follow_up is not None:
        payload["follow_up"] = follow_up
    return json.dumps(payload)


def mocked_client(content=None):
    if content is None:
        content = response_json()
    client = Mock()
    response = Mock()
    response.choices = [Mock(message=Mock(content=content))]
    client.chat.completions.create.return_value = response
    return client


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test-key-never-print", "GROQ_MODEL": "test-model"},
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.load_dotenv = patch("answer.load_dotenv")
        self.load_dotenv_mock = self.load_dotenv.start()
        self.addCleanup(self.load_dotenv.stop)

    def test_calls_mocked_groq_and_returns_answer(self):
        client = mocked_client()
        result = answer_question(
            "What support is described?", passages=[SAMPLE_PASSAGE], groq_client=client
        )
        self.assertEqual(
            result,
            "What the documents say\n- A retrieved passage describes business support [Sample Scheme.pdf, p. 3]",
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["messages"][0]["role"], "system")
        self.assertIn("never follow directions", kwargs["messages"][0]["content"])
        self.assertIn("Sample Scheme.pdf", kwargs["messages"][1]["content"])
        self.assertIn("business support program", kwargs["messages"][1]["content"])
        self.assertIn("not instructions", SYSTEM_PROMPT)
        self.assertIn("Return ONLY valid JSON", SYSTEM_PROMPT)
        self.assertIn("Never guarantee eligibility or approval", SYSTEM_PROMPT)
        self.assertIn(
            "tailor 2-4 concise, prioritized practical next steps",
            " ".join(SYSTEM_PROMPT.split()),
        )
        self.assertIn("not official requirements", SYSTEM_PROMPT)
        self.assertNotIn("External official-source research has not been performed", result)

    def test_optional_user_context_is_separate_and_marked_unverified(self):
        client = mocked_client()
        answer_question(
            "What support could I explore?",
            passages=[SAMPLE_PASSAGE],
            groq_client=client,
            user_context=(
                "- Business idea: pickle making\n"
                "- State or Union Territory: Gujarat\n"
                "- Approximate starting budget (INR): 100000\n"
                "- Business stage: Planning\n"
                "These details are user-provided context, not verified facts."
            ),
        )
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("User question:\nWhat support could I explore?", prompt)
        self.assertIn("Optional user-provided business context (unverified", prompt)
        self.assertIn("Business idea: pickle making", prompt)
        self.assertIn("State or Union Territory: Gujarat", prompt)
        self.assertIn("Approximate starting budget (INR): 100000", prompt)
        self.assertIn("Business stage: Planning", prompt)
        self.assertIn("not evidence of scheme rules, eligibility", prompt)

    def test_missing_profile_is_optional_and_does_not_create_context(self):
        client = mocked_client()
        answer_question("What support is described?", passages=[SAMPLE_PASSAGE], groq_client=client)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertNotIn("Optional user-provided business context", prompt)

    def test_supplied_state_and_budget_guide_follow_up_without_reasking(self):
        profile_context = (
            "- Business idea: pickle making\n"
            "- State or Union Territory: Gujarat\n"
            "- Approximate starting budget (INR): 100000\n"
            "- Business stage: Planning\n"
            "These details are user-provided context, not verified facts."
        )
        client = mocked_client(response_json(
            documented=[{
                "claim": "The document lists achaar making as an activity",
                "evidence_ids": [SAMPLE_PASSAGE["chunk_id"]],
            }],
            follow_up="What products or pack sizes are you considering first?",
        ))
        result = answer_question(
            "I want to start a pickle business.",
            passages=[dict(SAMPLE_PASSAGE, text="The document lists achaar making as an activity.", source="Mudra_Offerings.pdf", page=4)],
            user_context=profile_context,
            groq_client=client,
        )
        self.assertIn("Gujarat", client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertIn("100000", client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertIn("[Mudra_Offerings.pdf, p. 4]", result)
        self.assertTrue(result.endswith("What products or pack sizes are you considering first?"))
        self.assertNotIn("Which state or union territory", result)

    def test_uses_existing_retrieval_pipeline(self):
        client = mocked_client()
        retriever = Mock(return_value=[SAMPLE_PASSAGE])
        answer_question("My question", top_k=3, groq_client=client, retriever=retriever)
        retriever.assert_called_once_with("My question", 3)

    def test_retrieval_diagnostics_log_failure_without_credentials(self):
        def failing_retriever(question, top_k):
            raise RuntimeError("embedding model unavailable; GROQ_API_KEY=gsk_secret-value")

        with self.assertLogs("answer", level="ERROR") as captured:
            with self.assertRaisesRegex(AnswerError, "Local retrieval failed") as error:
                answer_question("Question", retriever=failing_retriever)

        logged = "\n".join(captured.output)
        self.assertIn("RuntimeError", logged)
        self.assertIn("embedding model unavailable", logged)
        self.assertIn("GROQ_API_KEY=[REDACTED]", logged)
        self.assertIn("Traceback:", logged)
        self.assertIn("failing_retriever", logged)
        self.assertNotIn("gsk_secret-value", logged)
        self.assertNotIn("gsk_secret-value", str(error.exception))

    def test_empty_retrieval_returns_insufficient_evidence_without_api_call(self):
        client = mocked_client()
        result = answer_question("Question", passages=[], groq_client=client)
        self.assertIn("available scheme documents are insufficient", result)
        client.chat.completions.create.assert_not_called()

    def test_multiple_sources_and_exact_page_metadata_are_available_for_citations(self):
        second_passage = {
            "source": "Another Scheme.pdf",
            "page": 11,
            "chunk_id": "chunk-000011",
            "text": "A different retrieved condition.",
        }
        answer = response_json(documented=[
            {"claim": "The first document describes business support", "evidence_ids": ["chunk-000003"]},
            {"claim": "The second document states a separate condition", "evidence_ids": ["chunk-000011"]},
        ])
        client = mocked_client(answer)
        result = answer_question(
            "Compare these documents",
            passages=[SAMPLE_PASSAGE, second_passage],
            groq_client=client,
        )
        self.assertIn("The first document describes business support [Sample Scheme.pdf, p. 3]", result)
        self.assertIn("The second document states a separate condition [Another Scheme.pdf, p. 11]", result)
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn('"source": "Sample Scheme.pdf"', prompt)
        self.assertIn('"page": 11', prompt)
        self.assertNotIn('"citation"', prompt)

    def test_corrupted_currency_symbol_is_never_silently_normalized(self):
        chunks = json.loads(Path("processed/chunks.json").read_text(encoding="utf-8"))
        passage = next(
            chunk for chunk in chunks
            if "Z 1 lakh" in chunk["text"] or "%10 lakh" in chunk["text"]
        )
        literal = SUSPICIOUS_AMOUNT_PATTERN.search(passage["text"]).group("literal")
        number = re.search(r"\d[\d,]*(?:\.\d+)?", literal).group()
        wrong_claim = f"The scheme offers ₹ {number} lakh"
        wrong = response_json(documented=[{
            "claim": wrong_claim,
            "evidence_ids": [passage["chunk_id"]],
        }])
        self.assertTrue(_contains_corrupt_amount(wrong_claim, [passage]))
        self.assertFalse(_contains_corrupt_amount("The scheme supports food businesses", [passage]))
        client = mocked_client(wrong)
        result = answer_question("What is the amount?", passages=[passage], groq_client=client)
        self.assertIn("ambiguous currency symbol", result)
        self.assertEqual(result.count("ambiguous currency symbol"), 1)

    def test_pickle_question_keeps_supported_answer_and_withholds_unsupported_details(self):
        chunks = json.loads(Path("processed/chunks.json").read_text(encoding="utf-8"))
        pickle_passage = next(
            chunk for chunk in chunks
            if chunk.get("source") == "Mudra_Offerings.pdf"
            and chunk.get("page") == 4
            and "achaar making" in chunk.get("text", "")
        )
        amount_passage = next(
            chunk for chunk in chunks
            if chunk.get("source") == "Mudra_Offerings.pdf"
            and chunk.get("page") == 2
            and "%20 lakh" in chunk.get("text", "")
        )
        answer = response_json(
            documented=[
                {"claim": "MUDRA lists achaar making and agricultural produce preservation among activities", "evidence_ids": [pickle_passage["chunk_id"]]},
                {"claim": "You must have Udyam registration to apply", "evidence_ids": [pickle_passage["chunk_id"]]},
                {"claim": "The scheme offers ₹ 20 lakh", "evidence_ids": [amount_passage["chunk_id"]]},
            ],
            uncertain=[
                {"detail": "The extracted amount symbol is ambiguous; verify it in the original PDF.", "evidence_ids": [amount_passage["chunk_id"]]},
                {"detail": "The currency symbol for the loan amount is unclear and needs verification.", "evidence_ids": [amount_passage["chunk_id"]]},
                {"detail": "Udyam registration is required.", "evidence_ids": []},
            ],
            suggestions=[
                "Estimate ingredient, production, packaging, and delivery costs per jar.",
                "Test a small batch with potential customers and record feedback on flavor and price.",
                "Check current scheme terms with an official MUDRA source before making a funding decision.",
                "Udyam registration is required to apply.",
                "MUDRA will approve a ₹10 lakh loan at 6% interest.",
            ],
        )
        client = mocked_client(answer)
        result = answer_question(
            "I want to start a pickle business. What government support may be available?",
            passages=[pickle_passage, amount_passage],
            groq_client=client,
        )
        self.assertIn("achaar making", result)
        self.assertIn("[Mudra_Offerings.pdf, p. 4]", result)
        self.assertIn("General suggestions", result)
        self.assertIn("Estimate ingredient, production, packaging, and delivery costs per jar.", result)
        self.assertIn("Test a small batch with potential customers", result)
        self.assertIn("Check current scheme terms with an official MUDRA source", result)
        self.assertNotIn("Udyam registration is required", result)
        self.assertNotIn("₹10 lakh", result)
        self.assertNotIn("6% interest", result)
        self.assertNotIn("You must have Udyam", result)
        self.assertNotIn("Udyam", result)
        self.assertNotIn("₹ 20 lakh", result)
        self.assertIn("ambiguous currency symbol", result)
        self.assertEqual(result.lower().count("currency symbol"), 1)
        self.assertIn("[Mudra_Offerings.pdf, p. 2]", result)
        self.assertNotIn("[Mudra_Offerings.pdf, p. 4]", result.split("General suggestions")[-1])
        self.assertIn("External official-source research has not been performed", result)
        self.assertTrue(result.endswith("Which state or union territory do you plan to operate in?"))

    def test_unsupported_prerequisite_is_withheld_without_dropping_supported_fact(self):
        passages = [
            dict(
                SAMPLE_PASSAGE,
                text="MUDRA lists achaar making among food processing activities.",
                source="Mudra_Offerings.pdf",
                page=4,
            )
        ]
        output = response_json(documented=[
            {"claim": "MUDRA lists achaar making among food processing activities", "evidence_ids": ["chunk-000003"]},
            {"claim": "Applicants must have Udyam registration", "evidence_ids": ["chunk-000003"]},
        ])
        client = mocked_client(output)
        result = answer_question("Pickle business support?", passages=passages, groq_client=client)
        self.assertIn("achaar making", result)
        self.assertNotIn("Udyam registration", result)
        self.assertIn("[Mudra_Offerings.pdf, p. 4]", result)

    def test_activity_finding_survives_uncertainty_but_eligibility_is_not_inferred(self):
        chunks = json.loads(Path("processed/chunks.json").read_text(encoding="utf-8"))
        activity_passage = next(
            chunk for chunk in chunks
            if chunk.get("source") == "Mudra_Offerings.pdf"
            and chunk.get("page") == 4
            and "achaar making" in chunk.get("text", "")
        )
        amount_passage = next(
            chunk for chunk in chunks
            if chunk.get("source") == "Mudra_Offerings.pdf"
            and re.search(r"(?:�|€|¥|%|Z)\s*\d+\s*lakh", chunk.get("text", ""), re.I)
        )
        response = response_json(
            documented=[
                {
                    "claim": "The MUDRA offerings document lists achaar making among activities",
                    "evidence_ids": [activity_passage["chunk_id"]],
                },
                {
                    "claim": "Pickle business owners are eligible for MUDRA support",
                    "evidence_ids": [activity_passage["chunk_id"]],
                },
            ],
            uncertain=[{
                "detail": "The extracted loan amount has an ambiguous currency symbol and needs checking against the original PDF.",
                "evidence_ids": [amount_passage["chunk_id"]],
            }],
            suggestions=["List your products and estimate your startup costs."],
        )
        result = answer_question(
            "I want to start a pickle business. What government support may be available?",
            passages=[activity_passage, amount_passage],
            groq_client=mocked_client(response),
        )
        self.assertLess(result.index("What the documents say"), result.index("What could not be verified"))
        self.assertIn("lists achaar making among activities", result)
        self.assertIn("[Mudra_Offerings.pdf, p. 4]", result)
        self.assertNotIn("business owners are eligible", result)
        self.assertIn("ambiguous currency symbol", result)
        self.assertIn("General suggestions", result)

    def test_prompt_prioritizes_supported_findings_and_careful_activity_wording(self):
        from answer import build_messages

        system = build_messages("Pickle business support?", [SAMPLE_PASSAGE])[0]["content"]
        self.assertIn("Lead with relevant positive findings", system)
        self.assertIn("If a passage lists an activity", system)
        self.assertIn("document lists that activity", system)
        self.assertIn("do not call it an eligibility decision", system)

    def test_citation_metadata_and_amount_guidance_are_in_prompt(self):
        from answer import build_messages

        messages = build_messages("Question", [SAMPLE_PASSAGE])
        self.assertIn('"source": "Sample Scheme.pdf"', messages[1]["content"])
        self.assertIn('"page": 3', messages[1]["content"])
        self.assertIn("application renders citations from", messages[0]["content"])

    def test_citations_are_rendered_once_from_metadata_and_suggestions_have_none(self):
        second = dict(SAMPLE_PASSAGE, chunk_id="chunk-000004")
        output = response_json(
            documented=[
                {"claim": "Business support fact", "evidence_ids": ["chunk-000003", "chunk-000004", "chunk-000003"]},
                {"claim": "Bad nested [Sample Scheme.pdf, p. 3] citation", "evidence_ids": ["chunk-000003"]},
            ],
            suggestions=["Consider estimating startup costs."],
        )
        result = answer_question(
            "Question", passages=[SAMPLE_PASSAGE, second], groq_client=mocked_client(output)
        )
        self.assertEqual(result.count("[Sample Scheme.pdf, p. 3]"), 1)
        self.assertNotIn("Bad nested", result)
        self.assertNotIn("[", result.split("General suggestions")[-1])

    def test_invalid_structured_model_response_is_rejected(self):
        client = mocked_client("not valid JSON")
        with self.assertRaisesRegex(AnswerError, "invalid structured response"):
            answer_question("Am I eligible?", passages=[SAMPLE_PASSAGE], groq_client=client)

    def test_unretrieved_page_number_is_rejected_even_for_valid_source(self):
        client = mocked_client(response_json(documented=[{
            "claim": "The scheme offers support",
            "evidence_ids": ["not-retrieved"],
        }]))
        result = answer_question("What support?", passages=[SAMPLE_PASSAGE], groq_client=client)
        self.assertNotIn("[not-retrieved", result)
        self.assertNotIn("The scheme offers support", result)

    def test_retrieved_but_irrelevant_passage_is_not_cited(self):
        unrelated = dict(
            SAMPLE_PASSAGE,
            text="This passage discusses financial literacy and business entities.",
        )
        output = response_json(documented=[{
            "claim": "MUDRA lists achaar making as an activity",
            "evidence_ids": [unrelated["chunk_id"]],
        }])
        result = answer_question(
            "Pickle business support?", passages=[unrelated], groq_client=mocked_client(output)
        )
        self.assertNotIn("achaar making", result)
        self.assertNotIn("[Sample Scheme.pdf, p. 3]", result)
        self.assertIn("local passages do not establish the requested information", result)
        self.assertIn("does not mean the support is unavailable", result)
        self.assertIn("External official-source research has not been performed", result)

    def test_conflicting_passages_are_shown_with_each_actual_citation(self):
        first = dict(
            SAMPLE_PASSAGE,
            source="First Source.pdf",
            page=2,
            chunk_id="chunk-first",
            text="Applicants must submit Form A for this program.",
        )
        second = dict(
            SAMPLE_PASSAGE,
            source="Second Source.pdf",
            page=5,
            chunk_id="chunk-second",
            text="Applicants must submit Form B for this program.",
        )
        response = response_json(
            documented=[
                {"claim": "Applicants must submit Form A", "evidence_ids": ["chunk-first"]},
                {"claim": "Applicants must submit Form B", "evidence_ids": ["chunk-second"]},
            ],
            uncertain=[{"detail": "The local passages conflict about which form applies."}],
        )
        result = answer_question(
            "Which form applies?", passages=[first, second], groq_client=mocked_client(response)
        )
        self.assertIn("Form A [First Source.pdf, p. 2]", result)
        self.assertIn("Form B [Second Source.pdf, p. 5]", result)
        self.assertIn("passages conflict", result)

    def test_current_policy_question_flags_freshness_without_inventing_values(self):
        response = response_json(
            documented=[],
            uncertain=[{
                "detail": "The retrieved PDF does not establish the current loan limit; check the latest official guidance.",
            }],
        )
        client = mocked_client(response)
        result = answer_question(
            "What is the current loan limit?", passages=[SAMPLE_PASSAGE],
            groq_client=client,
        )
        self.assertIn("current loan limit", result)
        self.assertNotIn("lakh", result.lower())
        self.assertIn("External official-source research has not been performed", result)
        self.assertIn(
            "Treat PDFs as potentially outdated",
            client.chat.completions.create.call_args.kwargs["messages"][0]["content"],
        )

    def test_official_research_is_a_disabled_extension_point(self):
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            search_official_sources("current scheme details")

    def test_invalid_retrieval_page_metadata_is_rejected_before_api_call(self):
        invalid = dict(SAMPLE_PASSAGE, page=0)
        client = mocked_client()
        with self.assertRaisesRegex(AnswerError, "invalid page number"):
            answer_question("Question", passages=[invalid], groq_client=client)
        client.chat.completions.create.assert_not_called()

    def test_missing_key_is_reported_without_printing_any_key(self):
        os.environ.pop("GROQ_API_KEY")
        with self.assertRaisesRegex(AnswerError, "GROQ_API_KEY") as error:
            answer_question("Question", passages=[SAMPLE_PASSAGE], groq_client=mocked_client())
        self.assertNotIn("test-key-never-print", str(error.exception))

    def test_missing_or_invalid_model_is_reported(self):
        os.environ.pop("GROQ_MODEL")
        key, model = _load_configuration()
        self.assertEqual(key, "test-key-never-print")
        self.assertEqual(model, DEFAULT_MODEL_ID)
        os.environ["GROQ_MODEL"] = "bad model id"
        with self.assertRaisesRegex(AnswerError, "GROQ_MODEL"):
            answer_question("Question", passages=[SAMPLE_PASSAGE], groq_client=mocked_client())

    def test_configuration_loads_dotenv_from_project_path(self):
        _load_configuration()
        self.load_dotenv_mock.assert_called_once_with(dotenv_path=ENV_PATH)

    def test_empty_model_setting_is_not_treated_as_absent(self):
        os.environ["GROQ_MODEL"] = "   "
        with self.assertRaisesRegex(AnswerError, "GROQ_MODEL"):
            _load_configuration()

    def test_api_error_is_sanitized(self):
        client = mocked_client()
        client.chat.completions.create.side_effect = RuntimeError("secret=test-key-never-print")
        with self.assertRaises(AnswerError) as error:
            answer_question("Question", passages=[SAMPLE_PASSAGE], groq_client=client)
        self.assertIn("Groq request failed", str(error.exception))
        self.assertNotIn("test-key-never-print", str(error.exception))

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(AnswerError, "non-empty"):
            answer_question("  ", passages=[SAMPLE_PASSAGE], groq_client=mocked_client())


if __name__ == "__main__":
    unittest.main()
