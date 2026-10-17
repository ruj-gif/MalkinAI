"""Tests for conservative memory updates and isolated persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from business_memory import (
    build_memory_context,
    build_retrieval_query,
    empty_business_memory,
    memory_from_json,
    memory_to_json,
    normalize_business_memory,
    record_assistant_suggestions,
    roll_conversation_history,
    update_memory_from_user,
)
from business_memory_store import delete_memory, generate_recovery_key, load_memory, save_memory


class BusinessMemoryTests(unittest.TestCase):
    def test_explicit_profile_facts_are_captured_and_new_values_replace_old(self):
        memory = update_memory_from_user(
            empty_business_memory(),
            "I want to start a pickle business in Gujarat. My budget is INR 80,000.",
        )
        facts = memory["user_provided"]
        self.assertEqual(facts["business_idea"], "pickle business")
        self.assertEqual(facts["state_or_location"], "Gujarat")
        self.assertEqual(facts["starting_budget"], {"amount": "80,000", "currency": "INR"})
        updated = update_memory_from_user(memory, "My budget is ₹120000 and I am based in Maharashtra.")
        self.assertEqual(updated["user_provided"]["starting_budget"]["amount"], "120000")
        self.assertEqual(updated["user_provided"]["state_or_location"], "Maharashtra")

    def test_missing_profile_fields_remain_empty_and_are_not_invented(self):
        memory = update_memory_from_user(empty_business_memory(), "Can you help me explore?")
        facts = memory["user_provided"]
        self.assertEqual(facts["business_idea"], "")
        self.assertEqual(facts["state_or_location"], "")
        self.assertEqual(facts["starting_budget"]["amount"], "")
        self.assertEqual(build_memory_context(memory), "")

    def test_amount_without_currency_does_not_assume_inr(self):
        memory = update_memory_from_user(empty_business_memory(), "My budget is 50000.")
        self.assertEqual(
            memory["user_provided"]["starting_budget"],
            {"amount": "50000", "currency": ""},
        )

    def test_assistant_suggestions_remain_separate_from_user_facts(self):
        memory = record_assistant_suggestions(
            empty_business_memory(),
            "General suggestions\n- Estimate your packaging costs.\n- Ask potential customers what they prefer.",
        )
        self.assertEqual(memory["user_provided"]["next_steps"], [])
        self.assertEqual(len(memory["assistant_suggestions"]), 2)
        context = build_memory_context(memory)
        self.assertEqual(context, "")

    def test_references_use_recent_turns_and_business_memory_for_retrieval(self):
        memory = update_memory_from_user(empty_business_memory(), "I want to start a pickle business.")
        history = [
            {"role": "user", "content": "I want to start a pickle business."},
            {"role": "assistant", "content": "The retrieved MUDRA passage lists achaar making."},
        ]
        query = build_retrieval_query("What about the cost of it?", history, memory)
        self.assertIn("What about the cost of it?", query)
        self.assertIn("pickle business", query)
        self.assertIn("achaar making", query)

    def test_long_history_moves_old_turns_to_bounded_summary(self):
        history = []
        for index in range(14):
            history.extend([
                {"role": "user", "content": f"Question {index}"},
                {"role": "assistant", "content": f"Answer {index}"},
            ])
        summary, recent = roll_conversation_history("", history)
        self.assertLessEqual(len(recent), 10)
        self.assertIn("Question 0", summary)
        self.assertIn("assistant: Answer 0", summary)
        self.assertLessEqual(len(summary), 1800)

    def test_manual_json_edit_is_normalized_and_bad_memory_fails_safely(self):
        memory = empty_business_memory()
        memory["user_provided"]["business_idea"] = "Handmade soaps"
        edited = memory_from_json(memory_to_json(memory))
        self.assertEqual(edited["user_provided"]["business_idea"], "Handmade soaps")
        self.assertEqual(normalize_business_memory({})["user_provided"]["business_idea"], "")
        with self.assertRaises(ValueError):
            memory_from_json("not json")


class BusinessMemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.database = Path(self.temp_dir.name) / "memory.sqlite3"
        self.addCleanup(self.temp_dir.cleanup)

    def test_profiles_are_isolated_and_can_be_saved_restored_edited_and_deleted(self):
        key_one, key_two = generate_recovery_key(), generate_recovery_key()
        self.assertNotEqual(key_one, key_two)
        first = empty_business_memory()
        first["user_provided"]["business_idea"] = "Pickle making"
        second = empty_business_memory()
        second["user_provided"]["business_idea"] = "Tailoring"

        save_memory(key_one, first, self.database)
        save_memory(key_two, second, self.database)
        restored = load_memory(key_one, self.database)
        self.assertEqual(restored["user_provided"]["business_idea"], "Pickle making")
        self.assertEqual(load_memory(key_two, self.database)["user_provided"]["business_idea"], "Tailoring")

        restored["user_provided"]["state_or_location"] = "Gujarat"
        save_memory(key_one, restored, self.database)
        self.assertEqual(load_memory(key_one, self.database)["user_provided"]["state_or_location"], "Gujarat")
        self.assertTrue(delete_memory(key_one, self.database))
        self.assertIsNone(load_memory(key_one, self.database))
        self.assertIsNotNone(load_memory(key_two, self.database))

    def test_recovery_key_is_not_stored_in_database(self):
        import sqlite3

        key = generate_recovery_key()
        save_memory(key, empty_business_memory(), self.database)
        connection = sqlite3.connect(self.database)
        try:
            stored = connection.execute("SELECT profile_key_hash FROM saved_business_profiles").fetchone()[0]
        finally:
            connection.close()
        self.assertNotEqual(stored, key)
        self.assertNotIn(key, stored)

    def test_corrupt_database_payload_is_treated_as_missing(self):
        import sqlite3

        key = generate_recovery_key()
        save_memory(key, empty_business_memory(), self.database)
        import hashlib
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
        connection = sqlite3.connect(self.database)
        try:
            connection.execute(
                "UPDATE saved_business_profiles SET memory_json = ? WHERE profile_key_hash = ?",
                (json.dumps(["corrupt"]), key_hash),
            )
            connection.commit()
        finally:
            connection.close()
        self.assertIsNone(load_memory(key, self.database))

    def test_invalid_database_file_is_treated_as_missing(self):
        key = generate_recovery_key()
        self.database.write_text("not a sqlite database", encoding="utf-8")
        self.assertIsNone(load_memory(key, self.database))


if __name__ == "__main__":
    unittest.main()
