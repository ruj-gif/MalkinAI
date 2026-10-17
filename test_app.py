"""Offline tests for Streamlit UI helpers and backend integration."""

import unittest

from app import (
    CUSTOM_CSS,
    MICROPHONE_MESSAGE,
    STARTER_PROMPTS,
    build_profile_context,
    activate_conversation,
    clear_profile,
    initialize_session_state,
    safe_error_message,
    submit_message,
    start_new_conversation,
)
from answer import AnswerError
from business_memory import empty_business_memory


class AppIntegrationTests(unittest.TestCase):
    def test_session_history_and_profile_are_initialized_without_resetting(self):
        state = {"messages": [{"role": "user", "content": "Earlier question"}], "profile_idea": "Tea shop"}
        initialize_session_state(state)
        self.assertEqual(state["messages"], [{"role": "user", "content": "Earlier question"}])
        self.assertEqual(state["profile_idea"], "Tea shop")
        self.assertEqual(state["profile_stage"], "Select a stage")
        self.assertEqual(state["messages"], state["conversations"][state["active_conversation_id"]])

    def test_new_conversation_has_separate_history_and_keeps_business_memory(self):
        state = {"messages": [{"role": "user", "content": "Old conversation"}]}
        initialize_session_state(state)
        old_id = state["active_conversation_id"]
        state["business_memory"]["user_provided"]["business_idea"] = "Pickle making"
        new_id = start_new_conversation(state)
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["conversations"][old_id][0]["content"], "Old conversation")
        self.assertEqual(state["business_memory"]["user_provided"]["business_idea"], "Pickle making")
        activate_conversation(state, old_id)
        self.assertEqual(state["messages"][0]["content"], "Old conversation")

    def test_clear_profile_resets_only_profile_fields(self):
        state = {
            "messages": [{"role": "assistant", "content": "Hello"}],
            "profile_idea": "Pickles",
            "business_memory": {
                "version": 1,
                "user_provided": {"business_idea": "Pickles"},
            },
        }
        initialize_session_state(state)
        clear_profile(state)
        self.assertEqual(state["profile_idea"], "")
        self.assertEqual(state["messages"], [{"role": "assistant", "content": "Hello"}])
        self.assertEqual(state["business_memory"]["user_provided"]["business_idea"], "")
        self.assertTrue(state["memory_history_backfill_done"])

    def test_profile_is_passed_separately_as_unverified_context(self):
        calls = []

        def backend(question, **kwargs):
            calls.append((question, kwargs))
            return "A supported statement [Scheme.pdf, p. 2]"

        history = []
        response = submit_message(
            history,
            STARTER_PROMPTS[0],
            {"idea": "Pickle making", "location": "Gujarat", "budget": "100000", "stage": "Planning", "language": "Hindi"},
            backend=backend,
        )
        self.assertEqual(calls[0][0], STARTER_PROMPTS[0])
        self.assertIn("Business idea: pickle business", calls[0][1]["user_context"])
        self.assertIn("State or Union Territory: Gujarat", calls[0][1]["user_context"])
        self.assertIn("Approximate starting budget (INR): 100000", calls[0][1]["user_context"])
        self.assertIn("Business stage: Planning", calls[0][1]["user_context"])
        self.assertIn("not verified facts", calls[0][1]["user_context"])
        self.assertNotIn("Hindi", calls[0][1]["user_context"])
        self.assertEqual(response["content"], "A supported statement [Scheme.pdf, p. 2]")
        self.assertEqual([message["role"] for message in history], ["user", "assistant"])
        self.assertIn("[Scheme.pdf, p. 2]", history[1]["content"])

    def test_follow_up_receives_previous_turns_and_updates_structured_memory(self):
        calls = []

        def backend(question, **kwargs):
            calls.append((question, kwargs))
            return "A useful next step."

        history = [
            {"role": "user", "content": "I want to start a pickle business."},
            {"role": "assistant", "content": "The retrieved MUDRA passage lists achaar making. [Mudra_Offerings.pdf, p. 4]"},
        ]
        memory = empty_business_memory()
        state = {"conversation_summaries": {}, "active_conversation_id": "conv-1"}
        submit_message(
            history,
            "What about the cost of it?",
            {},
            backend=backend,
            business_memory=memory,
            conversation_state=state,
        )
        self.assertEqual(calls[0][0], "What about the cost of it?")
        self.assertEqual(len(calls[0][1]["conversation_history"]), 2)
        self.assertIn("pickle business", calls[0][1]["business_memory"]["user_provided"]["business_idea"])
        self.assertIn("cost of it", history[-2]["content"])

    def test_new_explicit_profile_values_override_old_memory(self):
        memory = empty_business_memory()
        memory["user_provided"]["state_or_location"] = "Gujarat"
        memory["user_provided"]["starting_budget"] = {"amount": "50000", "currency": "INR"}
        # Capture the current memory supplied to the backend to verify current-turn facts win.
        captured = {}
        def capture(question, **kwargs):
            captured.update(kwargs)
            return "Done."
        submit_message([], "I moved to Maharashtra and my budget is INR 90000.", {}, backend=capture, business_memory=memory)
        updated = captured["business_memory"]["user_provided"]
        self.assertEqual(updated["state_or_location"], "Maharashtra")
        self.assertEqual(updated["starting_budget"]["amount"], "90000")

    def test_official_source_verification_is_optional_and_forwarded(self):
        calls = []

        def backend(question, **kwargs):
            calls.append(kwargs)
            return "Answer"

        submit_message(
            [],
            "What does MUDRA support?",
            {},
            backend=backend,
            official_research=True,
        )
        self.assertTrue(calls[0]["official_research"])

    def test_backend_error_is_safe_and_keeps_turn_in_history(self):
        def backend(question, **kwargs):
            raise AnswerError("Groq request failed. secret=never-show")

        history = []
        response = submit_message(history, "Question", {}, backend=backend)
        self.assertTrue(response["error"])
        self.assertIn("couldn’t reach the answer service", response["content"])
        self.assertNotIn("never-show", response["content"])
        self.assertEqual(len(history), 2)

    def test_retrieval_failure_gets_a_clear_safe_message(self):
        message = safe_error_message(AnswerError("Local retrieval failed: private path"))
        self.assertIn("couldn’t search the local scheme documents", message)
        self.assertNotIn("private path", message)

    def test_profile_empty_and_microphone_placeholder_are_transparent(self):
        self.assertEqual(build_profile_context({}), "")
        self.assertEqual(build_profile_context({"stage": "Select a stage"}), "")
        self.assertIn("Voice input is coming soon", MICROPHONE_MESSAGE)
        self.assertIn("not recording audio", MICROPHONE_MESSAGE)

    def test_theme_styles_sidebar_contrast_top_spacing_and_mobile_layout(self):
        self.assertIn('[data-testid="stSidebar"] p', CUSTOM_CSS)
        self.assertIn("color: var(--plum) !important", CUSTOM_CSS)
        self.assertIn('[data-testid="stMainBlockContainer"] { padding-top: 1.5rem', CUSTOM_CSS)
        self.assertIn("@media (max-width: 700px)", CUSTOM_CSS)
        self.assertIn("flex-wrap: wrap", CUSTOM_CSS)
        self.assertIn('[class*="st-key-malkin-assistant-message-"] li::marker', CUSTOM_CSS)
        self.assertIn('[class*="st-key-malkin-user-message-"]', CUSTOM_CSS)
        self.assertIn('.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInput"] > div', CUSTOM_CSS)
        self.assertIn('.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputTextArea"]', CUSTOM_CSS)
        self.assertIn('.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputSubmitButton"]', CUSTOM_CSS)
        self.assertIn('.st-key-voice_button button {', CUSTOM_CSS)
        self.assertIn('.st-key-voice_button button:hover:not(:disabled)', CUSTOM_CSS)
        self.assertIn('.st-key-voice_button button:focus-visible', CUSTOM_CSS)
        self.assertIn('.st-key-voice_button button:disabled', CUSTOM_CSS)
        self.assertIn('background: #FFF9F4 !important', CUSTOM_CSS)
        self.assertIn('background: #FFFDFB !important', CUSTOM_CSS)
        self.assertIn('background: #F7B7A7 !important', CUSTOM_CSS)
        self.assertIn('color: #48223F !important', CUSTOM_CSS)
        self.assertIn('Voice input is coming soon', MICROPHONE_MESSAGE)
        self.assertNotIn('\n[data-testid="stChatInput"]', CUSTOM_CSS)

    def test_starter_prompts_are_real_questions_not_prebuilt_answers(self):
        self.assertEqual(len(STARTER_PROMPTS), 4)
        self.assertTrue(all(prompt.endswith("?") or prompt.endswith(".") for prompt in STARTER_PROMPTS))


if __name__ == "__main__":
    unittest.main()
