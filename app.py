"""Streamlit chat interface for the existing Malkin AI RAG backend."""

from __future__ import annotations

import sqlite3
import uuid
import logging
from typing import Any, Callable, MutableMapping

from answer import (
    AnswerError,
    _safe_diagnostic_detail,
    _safe_diagnostic_traceback,
    answer_question,
)
from business_memory import (
    build_memory_context,
    empty_business_memory,
    memory_from_json,
    memory_to_json,
    merge_profile_into_memory,
    normalize_business_memory,
    record_assistant_suggestions,
    roll_conversation_history,
    update_memory_from_user,
)
from business_memory_store import (
    DEFAULT_DATABASE_PATH,
    delete_memory,
    generate_recovery_key,
    load_memory,
    save_memory,
)

try:
    import streamlit as st
except ImportError:  # Keep helpers testable when optional UI dependencies are absent.
    st = None


STARTER_PROMPTS = (
    "I want to start a pickle business.",
    "What funding options could I explore?",
    "Help me plan a small business with a limited budget.",
    "What information do you need to recommend schemes?",
)
LOGGER = logging.getLogger(__name__)
MICROPHONE_MESSAGE = "Voice input is coming soon. We are not recording audio yet."
PROFILE_DEFAULTS = {
    "profile_idea": "",
    "profile_location": "",
    "profile_budget": "",
    "profile_stage": "Select a stage",
    "profile_language": "English",
}


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Set initial per-conversation and business-memory state without resetting it."""
    if "conversations" not in state:
        conversation_id = uuid.uuid4().hex
        state["conversations"] = {conversation_id: list(state.get("messages", []))}
        state["active_conversation_id"] = conversation_id
        state["conversation_summaries"] = {conversation_id: state.get("conversation_summary", "")}
    state.setdefault("active_conversation_id", next(iter(state["conversations"]), uuid.uuid4().hex))
    state.setdefault("conversation_summaries", {})
    active_id = state["active_conversation_id"]
    state["conversations"].setdefault(active_id, list(state.get("messages", [])))
    state["conversation_summaries"].setdefault(active_id, "")
    # Preserve the existing app/test key as an alias of the active conversation.
    state["messages"] = state["conversations"][active_id]
    state["conversation_summary"] = state["conversation_summaries"][active_id]
    state.setdefault("conversation_select", active_id)
    state["business_memory"] = normalize_business_memory(state.get("business_memory"))
    state.setdefault("memory_persistence_enabled", False)
    state.setdefault("memory_recovery_key", None)
    state.setdefault("memory_history_backfill_done", False)
    for key, value in PROFILE_DEFAULTS.items():
        state.setdefault(key, value)


def clear_profile(state: MutableMapping[str, Any]) -> None:
    """Clear profile and structured business facts, leaving conversation history intact."""
    for key, value in PROFILE_DEFAULTS.items():
        state[key] = value
    state["business_memory"] = empty_business_memory()
    state["refresh_memory_editor"] = True
    state["memory_history_backfill_done"] = True


def start_new_conversation(state: MutableMapping[str, Any]) -> str:
    """Create an isolated conversation while preserving the user's business memory."""
    conversation_id = uuid.uuid4().hex
    state.setdefault("conversations", {})[conversation_id] = []
    state.setdefault("conversation_summaries", {})[conversation_id] = ""
    state["active_conversation_id"] = conversation_id
    state["conversation_select"] = conversation_id
    state["messages"] = state["conversations"][conversation_id]
    state["conversation_summary"] = ""
    return conversation_id


def activate_conversation(state: MutableMapping[str, Any], conversation_id: str) -> None:
    """Switch the active view to an existing session-local conversation."""
    if conversation_id not in state.get("conversations", {}):
        raise ValueError("Conversation was not found in this session.")
    state["active_conversation_id"] = conversation_id
    state["messages"] = state["conversations"][conversation_id]
    state["conversation_summary"] = state.get("conversation_summaries", {}).get(conversation_id, "")


def build_profile_context(profile: dict[str, Any], memory: dict[str, Any] | None = None) -> str:
    """Format optional profile details as explicitly unverified user context."""
    facts = normalize_business_memory(memory)["user_provided"] if memory else {}
    budget = facts.get("starting_budget", {})
    budget_value = budget.get("amount") or profile.get("budget")
    budget_currency = budget.get("currency") or ("INR" if profile.get("budget") else "")
    budget_label = (
        f"Approximate starting budget ({budget_currency})" if budget_currency
        else "Approximate starting budget (currency not stated)"
    )
    fields = (
        ("Business idea", facts.get("business_idea") or profile.get("idea")),
        ("State or Union Territory", facts.get("state_or_location") or profile.get("location")),
        (budget_label, budget_value),
        (
            "Business stage",
            facts.get("business_stage") or (
                profile.get("stage") if profile.get("stage") != "Select a stage" else ""
            ),
        ),
    )
    lines = [f"- {label}: {str(value).strip()}" for label, value in fields if value and str(value).strip()]
    if not lines:
        return ""
    return "\n".join(lines) + "\nThese details are user-provided context, not verified facts or official eligibility evidence."


def safe_error_message(error: Exception) -> str:
    """Map backend exceptions to useful UI errors without exposing internals."""
    message = str(error).casefold()
    if "groq_api_key" in message:
        return (
            "The answer service is not configured yet. Check the local .env setup or "
            "GROQ_API_KEY in Streamlit Cloud Secrets."
        )
    if "retrieval" in message or "embedding index" in message:
        return "I couldn’t search the local scheme documents. Please check the processed files and try again."
    if "groq request" in message or "groq" in message or "network" in message:
        return "I couldn’t reach the answer service. Please check the connection and try again."
    return "I couldn’t complete that response. Please try again."


def submit_message(
    history: list[dict[str, Any]],
    question: str,
    profile: dict[str, Any],
    backend: Callable[..., str] = answer_question,
    official_research: bool = False,
    business_memory: dict[str, Any] | None = None,
    conversation_summary: str = "",
    conversation_state: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append a user turn and its backend response to the current session history."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    summary, recent_history = roll_conversation_history(conversation_summary, history)
    history[:] = recent_history
    updated_memory = normalize_business_memory(business_memory)
    backfill_history = (
        conversation_state is None
        or not conversation_state.get("memory_history_backfill_done", False)
    )
    if backfill_history:
        for prior_message in recent_history:
            if isinstance(prior_message, dict) and prior_message.get("role") == "user":
                updated_memory = update_memory_from_user(updated_memory, prior_message.get("content", ""))
    updated_memory = merge_profile_into_memory(updated_memory, profile)
    updated_memory = update_memory_from_user(updated_memory, question)
    context = build_profile_context(profile, updated_memory)
    previous_turns = list(history)
    history.append({"role": "user", "content": question})
    if conversation_state is not None:
        conversation_state["business_memory"] = updated_memory
        conversation_state["refresh_memory_editor"] = True
        conversation_state["memory_history_backfill_done"] = True
        conversation_state["conversation_summary"] = summary
        active_id = conversation_state.get("active_conversation_id")
        if active_id in conversation_state.get("conversation_summaries", {}):
            conversation_state["conversation_summaries"][active_id] = summary
    try:
        backend_options: dict[str, Any] = {
            "user_context": context or None,
            "conversation_history": previous_turns,
            "conversation_summary": summary or None,
            "business_memory": updated_memory,
        }
        if official_research:
            backend_options["official_research"] = True
        answer = backend(question, **backend_options)
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerError("The answer service returned an empty response.")
        assistant_message = {"role": "assistant", "content": answer.strip()}
    except Exception as exc:  # UI boundary: never surface stack traces or raw payloads.
        if not isinstance(exc, AnswerError):
            LOGGER.error(
                "Unexpected answer pipeline failure (%s): %s\nTraceback:\n%s",
                type(exc).__name__,
                _safe_diagnostic_detail(exc),
                _safe_diagnostic_traceback(exc),
            )
        assistant_message = {
            "role": "assistant",
            "content": safe_error_message(exc),
            "error": True,
        }
    history.append(assistant_message)
    updated_memory = record_assistant_suggestions(updated_memory, assistant_message["content"])
    if conversation_state is not None:
        conversation_state["business_memory"] = updated_memory
    return assistant_message


def _restore_memory_before_widgets(state: MutableMapping[str, Any], recovery_key: str) -> bool:
    """Restore an opted-in, key-isolated profile and sync the existing sidebar fields."""
    try:
        restored = load_memory(recovery_key)
    except (OSError, ValueError, sqlite3.Error):
        restored = None
    if restored is None:
        state["memory_notice"] = "No saved memory was found for that recovery key."
        return False
    state["business_memory"] = restored
    state["memory_history_backfill_done"] = True
    state["memory_recovery_key"] = recovery_key
    state["memory_persistence_enabled"] = True
    facts = restored["user_provided"]
    state["profile_idea"] = facts.get("business_idea", "")
    state["profile_location"] = facts.get("state_or_location", "")
    state["profile_budget"] = facts.get("starting_budget", {}).get("amount", "")
    state["profile_stage"] = facts.get("business_stage") or "Select a stage"
    state["memory_editor_text"] = memory_to_json(restored)
    state["memory_notice"] = "Business memory restored for this session."
    return True


def _delete_saved_memory_callback() -> None:
    """Delete saved memory from the button callback before its checkbox rerenders."""
    key = st.session_state.get("memory_recovery_key")
    try:
        deleted = delete_memory(key) if key else False
    except (OSError, ValueError, sqlite3.Error):
        deleted = False
    st.session_state["memory_recovery_key"] = None
    st.session_state["memory_persistence_enabled"] = False
    st.session_state["memory_notice"] = (
        "Saved business memory deleted." if deleted else "There was no saved memory to delete."
    )


CUSTOM_CSS = """
<style>
:root { --plum: #43213d; --coral: #cf5c50; --peach: #fff0e7; --cream: #fffaf5; }
* { box-sizing: border-box; }
.stApp { background: var(--cream); color: var(--plum); }
.block-container { width: 100%; max-width: 1080px; padding-bottom: 2rem; margin-top: 0 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem !important; margin-top: 0 !important; }
header[data-testid="stHeader"] { background: var(--cream); }
header[data-testid="stHeader"] button, header[data-testid="stHeader"] [data-testid="stToolbar"] { color: var(--plum); }
[data-testid="stSidebar"] { background: #f8eee8; border-right: 1px solid #eadbd3; color: var(--plum); }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--plum) !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li { color: var(--plum) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div { background: #fff !important; color: var(--plum) !important; }
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder { color: #715c69 !important; opacity: 1; }
[data-baseweb="popover"] [role="option"] { color: var(--plum) !important; }
.block-container h1, .block-container h2, .block-container h3 { color: var(--plum); letter-spacing: -0.025em; }
.malkin-brand { color: var(--plum); font-size: 1.35rem; font-weight: 750; letter-spacing: -.03em; }
.malkin-tagline { color: #725d6b; font-size: .9rem; margin-top: .15rem; }
.welcome-card { background: white; border: 1px solid #efdfd7; border-radius: 22px; padding: clamp(1.4rem, 4vw, 2.8rem); margin: .75rem 0 1.2rem; box-shadow: 0 8px 26px rgba(67,33,61,.045); }
.welcome-eyebrow { color: var(--coral); font-size: .78rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; }
.welcome-card h1 { margin: .6rem 0 .7rem; font-size: clamp(2rem, 5vw, 3rem); line-height: 1.08; }
.welcome-card p { max-width: 650px; color: #655461; font-size: 1.04rem; line-height: 1.6; }
.stButton > button { border-radius: 13px; min-height: 2.8rem; border: 1px solid #ead9d2; color: var(--plum) !important; background: white; text-align: left; white-space: normal; overflow-wrap: anywhere; transition: border-color .15s ease, transform .15s ease; }
.stButton > button * { color: var(--plum) !important; }
.stButton > button:hover { border-color: var(--coral); color: var(--plum) !important; transform: translateY(-1px); }
/* The keyed containers below are app-owned wrappers around each chat bubble. */
[class*="st-key-malkin-user-message-"] { color: var(--plum) !important; background: var(--peach); border: 1px solid #efd9cd; border-radius: 18px; padding: .9rem 1rem; }
[class*="st-key-malkin-assistant-message-"] { color: #332a31 !important; background: white; border: 1px solid #efe4de; border-radius: 18px; padding: .9rem 1rem; }
[class*="st-key-malkin-user-message-"] p,
[class*="st-key-malkin-user-message-"] h1,
[class*="st-key-malkin-user-message-"] h2,
[class*="st-key-malkin-user-message-"] h3,
[class*="st-key-malkin-user-message-"] h4,
[class*="st-key-malkin-user-message-"] h5,
[class*="st-key-malkin-user-message-"] h6,
[class*="st-key-malkin-user-message-"] li,
[class*="st-key-malkin-user-message-"] ul,
[class*="st-key-malkin-user-message-"] ol,
[class*="st-key-malkin-user-message-"] strong,
[class*="st-key-malkin-user-message-"] em,
[class*="st-key-malkin-user-message-"] a,
[class*="st-key-malkin-user-message-"] code { color: var(--plum) !important; overflow-wrap: anywhere; }
[class*="st-key-malkin-assistant-message-"] p,
[class*="st-key-malkin-assistant-message-"] h1,
[class*="st-key-malkin-assistant-message-"] h2,
[class*="st-key-malkin-assistant-message-"] h3,
[class*="st-key-malkin-assistant-message-"] h4,
[class*="st-key-malkin-assistant-message-"] h5,
[class*="st-key-malkin-assistant-message-"] h6,
[class*="st-key-malkin-assistant-message-"] li,
[class*="st-key-malkin-assistant-message-"] ul,
[class*="st-key-malkin-assistant-message-"] ol,
[class*="st-key-malkin-assistant-message-"] strong,
[class*="st-key-malkin-assistant-message-"] em,
[class*="st-key-malkin-assistant-message-"] blockquote,
[class*="st-key-malkin-assistant-message-"] code,
[class*="st-key-malkin-assistant-message-"] a { color: #332a31 !important; overflow-wrap: anywhere; line-height: 1.6; }
[class*="st-key-malkin-assistant-message-"] li::marker { color: var(--plum) !important; }
[class*="st-key-malkin-user-message-"] li::marker { color: var(--plum) !important; }
[class*="st-key-malkin-assistant-message-"] code { background: #f5eee9; }
[class*="st-key-malkin-assistant-message-"] a { text-decoration: underline; text-decoration-color: var(--coral); }
[class*="st-key-malkin-assistant-message-"] a:hover { color: var(--coral) !important; }
[data-testid="stChatMessage"] { background: transparent; border: 0; padding: .3rem 0 .7rem; }
.st-key-malkin-composer { background: transparent; color: var(--plum); }
.st-key-malkin-composer [data-testid="stHorizontalBlock"] { align-items: center; flex-wrap: nowrap; gap: .6rem; }
.st-key-malkin-composer [data-testid="column"] { min-width: 0; }
/* Streamlit 1.66 renders a native chat_input with a dark themed inner panel. */
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInput"] {
  padding: 0;
  border: 0;
  border-radius: 16px;
  background: transparent;
  box-shadow: none;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInput"] > div {
  border: 1px solid #F1D9CF !important;
  border-radius: 16px !important;
  background: #FFF9F4 !important;
  box-shadow: none !important;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputTextArea"] {
  min-height: 3.15rem;
  border-radius: 12px !important;
  background: #FFFDFB !important;
  color: #48223F !important;
  caret-color: #CF5C50;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputTextArea"]::placeholder {
  color: #8B7185 !important;
  opacity: 1;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputSubmitButton"] {
  min-width: 2.55rem;
  min-height: 2.55rem;
  margin: .25rem;
  border: 1px solid #CF5C50 !important;
  border-radius: 12px !important;
  background: #F7B7A7 !important;
  color: #48223F !important;
  transition: background-color .15s ease, border-color .15s ease, transform .15s ease;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputSubmitButton"]:hover:not(:disabled) {
  border-color: #A8423B !important;
  background: #EF9F8D !important;
  color: #32162F !important;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputSubmitButton"]:focus-visible {
  outline: 3px solid #A8423B !important;
  outline-offset: 2px;
}
.st-key-malkin-composer .st-key-malkin-chat-input [data-testid="stChatInputSubmitButton"] svg { color: #48223F !important; }
.st-key-malkin-composer .st-key-voice_button { display: flex; align-items: center; height: 100%; }
.st-key-voice_button button {
  min-height: 3.15rem;
  border: 1px solid #CF5C50 !important;
  border-radius: 14px !important;
  background: #F7B7A7 !important;
  color: #48223F !important;
  text-align: center;
  white-space: nowrap;
  transition: background-color .15s ease, border-color .15s ease, box-shadow .15s ease;
}
.st-key-voice_button button * { color: #48223F !important; }
.st-key-voice_button button:hover:not(:disabled) {
  border-color: #A8423B !important;
  background: #EF9F8D !important;
  color: #32162F !important;
}
.st-key-voice_button button:focus-visible {
  outline: 3px solid #A8423B !important;
  outline-offset: 2px;
  box-shadow: 0 0 0 2px #FFF9F4 !important;
}
.st-key-voice_button button:disabled {
  border-color: #D6BEB6 !important;
  background: #F7DDD5 !important;
  color: #715C69 !important;
  opacity: 1;
  cursor: not-allowed;
}
.st-key-voice_button button:disabled * { color: #715C69 !important; }
@media (max-width: 700px) {
  [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 1.5rem !important; }
  .welcome-card { border-radius: 17px; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: .55rem; }
  [data-testid="stHorizontalBlock"] > [data-testid="column"] { flex: 1 1 100%; min-width: 0; }
  .st-key-malkin-composer [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
  .st-key-malkin-composer [data-testid="column"]:first-child { flex: 0 0 5.25rem !important; }
  .st-key-malkin-composer [data-testid="column"]:last-child { flex: 1 1 auto !important; }
}
</style>
"""


def _profile_from_state(state: MutableMapping[str, Any]) -> dict[str, Any]:
    return {
        "idea": state["profile_idea"],
        "location": state["profile_location"],
        "budget": state["profile_budget"],
        "stage": state["profile_stage"],
        "language": state["profile_language"],
    }


def _reset_profile() -> None:
    clear_profile(st.session_state)


def _render_message(message: dict[str, Any], index: int) -> None:
    with st.chat_message(message["role"]):
        role_class = "user" if message["role"] == "user" else "assistant"
        with st.container(key=f"malkin-{role_class}-message-{index}"):
            if message.get("error"):
                st.error(message["content"])
            else:
                # Markdown is rendered with Streamlit's default HTML escaping.
                st.markdown(message["content"])


def main() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install project dependencies with "
            "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt'."
        )
    st.set_page_config(page_title="Malkin AI", page_icon="🌺", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    initialize_session_state(st.session_state)
    if st.session_state.pop("refresh_memory_editor", False):
        st.session_state["memory_editor_text"] = memory_to_json(st.session_state.business_memory)

    with st.sidebar:
        st.markdown('<div class="malkin-brand">🌺 Malkin AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="malkin-tagline">Your idea. Your business. Your next step.</div>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigate", ["Chat", "About Malkin AI"], label_visibility="collapsed")

        if st.button("＋ New conversation", key="new_conversation_button", width="stretch"):
            start_new_conversation(st.session_state)
            st.rerun()
        conversation_ids = list(st.session_state.conversations)
        selected_conversation = st.selectbox(
            "Active conversation",
            options=conversation_ids,
            key="conversation_select",
            format_func=lambda conversation_id: f"Conversation {conversation_ids.index(conversation_id) + 1}",
        )
        activate_conversation(st.session_state, selected_conversation)

        with st.expander("Business memory & privacy", expanded=False):
            st.caption(
                "Memory is separate from chat history. It stores business facts you provide; "
                "assistant suggestions and PDF facts are not promoted into user facts."
            )
            st.caption(
                "Saved memory is not encrypted in the local SQLite file. Avoid putting "
                "confidential information in it. On hosted deployments, the file may not "
                "survive a rebuild unless persistent storage is configured."
            )
            with st.form("restore_business_memory_form"):
                restore_key = st.text_input(
                    "Restore with recovery key", type="password", key="restore_memory_key_input"
                )
                restore_clicked = st.form_submit_button("Restore saved memory")
            if restore_clicked:
                _restore_memory_before_widgets(st.session_state, restore_key)

            st.session_state.setdefault("memory_editor_text", memory_to_json(st.session_state.business_memory))
            with st.form("edit_business_memory_form"):
                st.text_area(
                    "View or edit structured memory (JSON)",
                    key="memory_editor_text",
                    height=260,
                )
                edit_clicked = st.form_submit_button("Apply memory edits")
            if edit_clicked:
                try:
                    st.session_state.business_memory = memory_from_json(
                        st.session_state.memory_editor_text
                    )
                    st.session_state.memory_history_backfill_done = True
                    edited_facts = st.session_state.business_memory["user_provided"]
                    st.session_state.profile_idea = edited_facts.get("business_idea", "")
                    st.session_state.profile_location = edited_facts.get("state_or_location", "")
                    st.session_state.profile_budget = edited_facts.get("starting_budget", {}).get("amount", "")
                    st.session_state.profile_stage = edited_facts.get("business_stage") or "Select a stage"
                    if st.session_state.get("memory_persistence_enabled") and st.session_state.get("memory_recovery_key"):
                        save_memory(
                            st.session_state.memory_recovery_key,
                            st.session_state.business_memory,
                        )
                    st.session_state.memory_notice = "Business memory updated."
                except (ValueError, OSError, TypeError, sqlite3.Error) as exc:
                    st.session_state.memory_notice = str(exc)

            st.json(st.session_state.business_memory)

            if st.button(
                "Delete saved memory",
                key="delete_saved_memory_button",
                on_click=_delete_saved_memory_callback,
                width="stretch",
                disabled=not bool(st.session_state.get("memory_recovery_key")),
            ):
                pass

            save_memory_opt_in = st.checkbox(
                "Consent to save memory on this app instance",
                key="memory_persistence_enabled",
                help=(
                    "Saves only structured business memory to a local SQLite file. "
                    "It does not save chat transcripts or API secrets. The recovery key "
                    "is required to restore or delete this profile."
                ),
            )
            if save_memory_opt_in:
                if not st.session_state.get("memory_recovery_key"):
                    st.session_state.memory_recovery_key = generate_recovery_key()
                persisted_memory = merge_profile_into_memory(
                    st.session_state.business_memory,
                    {
                        "idea": st.session_state.get("profile_idea", ""),
                        "location": st.session_state.get("profile_location", ""),
                        "budget": st.session_state.get("profile_budget", ""),
                        "stage": st.session_state.get("profile_stage", ""),
                    },
                )
                st.session_state.business_memory = persisted_memory
                st.session_state.refresh_memory_editor = True
                try:
                    save_memory(st.session_state.memory_recovery_key, persisted_memory)
                    st.caption(f"Saved locally in SQLite: {DEFAULT_DATABASE_PATH.name}")
                    st.caption("Keep this recovery key private. Anyone with it can restore this memory.")
                    st.code(st.session_state.memory_recovery_key, language=None)
                    st.download_button(
                        "Download recovery key",
                        data=st.session_state.memory_recovery_key,
                        file_name="malkin_business_memory_recovery_key.txt",
                        mime="text/plain",
                        width="stretch",
                    )
                except (OSError, ValueError, sqlite3.Error) as exc:
                    st.warning(f"Memory could not be saved: {exc}")
            if notice := st.session_state.pop("memory_notice", None):
                st.info(notice)

        st.markdown("## Your business profile")
        st.caption("Optional context helps personalize the conversation. It is not verified scheme information.")
        st.text_input("Business idea", placeholder="e.g. Pickle making", key="profile_idea")
        st.text_input("State or Union Territory", placeholder="e.g. Gujarat", key="profile_location")
        st.text_input("Approximate starting budget in INR", placeholder="e.g. 100000", key="profile_budget")
        st.selectbox(
            "Business stage",
            ["Select a stage", "Exploring", "Planning", "Already operating"],
            key="profile_stage",
        )
        st.selectbox("Preferred language", ["English", "Hindi", "Gujarati", "Other"], key="profile_language")
        st.caption("Language preference is saved for future multilingual support; responses currently remain in English.")
        st.checkbox(
            "Verify with official sources online",
            key="official_research_enabled",
            help=(
                "When enabled, Malkin AI fetches relevant pages from the Department "
                "of Financial Services and FSSAI/FoSCoS. This does not guarantee loan approval."
            ),
        )
        st.caption("Optional online checking uses official pages only; uploaded-PDF citations remain separate.")
        st.button("Clear profile", on_click=_reset_profile, use_container_width=True)

    if page == "About Malkin AI":
        st.title("About Malkin AI")
        st.write("Malkin AI helps women entrepreneurs explore business ideas and understand possible government support using the project’s local scheme documents.")
        st.info("Document answers include source citations when the retrieved passages support them. Current policy details may need verification against official sources.")
        return

    st.markdown('<div class="malkin-brand">🌺 Malkin AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="malkin-tagline">Your idea. Your business. Your next step.</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            '<div class="welcome-card"><div class="welcome-eyebrow">A thoughtful place to begin</div>'
            '<h1>Let’s build your business together.</h1>'
            '<p>Explore a business idea, learn what your local scheme documents say about possible support, and choose a practical next step. You can ask a question in your own words.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Try asking**")
        starter_columns = st.columns(2)
        selected_question = None
        for index, prompt in enumerate(STARTER_PROMPTS):
            with starter_columns[index % 2]:
                if st.button(prompt, key=f"starter_{index}", use_container_width=True):
                    selected_question = prompt
    else:
        selected_question = None
        if st.session_state.conversation_summary:
            with st.expander("Earlier conversation summary", expanded=False):
                st.caption(st.session_state.conversation_summary)
        for index, message in enumerate(st.session_state.messages):
            _render_message(message, index)

    if st.session_state.pop("voice_notice", False):
        st.info(MICROPHONE_MESSAGE)

    with st.container(key="malkin-composer"):
        voice_col, input_col = st.columns([0.2, 0.8])
        with voice_col:
            with st.container(key="voice_button"):
                if st.button("🎙 Voice", help="Voice input placeholder", key="malkin-voice-placeholder", use_container_width=True):
                    st.session_state.voice_notice = True
                    st.rerun()
        with input_col:
            typed_question = st.chat_input(
                "Tell me what you’re hoping to build…",
                key="malkin-chat-input",
            )

    question = typed_question or selected_question
    if question:
        profile = _profile_from_state(st.session_state)
        message_index = len(st.session_state.messages)
        _render_message({"role": "user", "content": question}, message_index)
        with st.chat_message("assistant"):
            with st.container(key=f"malkin-assistant-message-{message_index + 1}"):
                with st.spinner("Looking through your question and preparing a response…"):
                    assistant_message = submit_message(
                        st.session_state.messages,
                        question,
                        profile,
                        official_research=st.session_state.get("official_research_enabled", False),
                        business_memory=st.session_state.business_memory,
                        conversation_summary=st.session_state.conversation_summary,
                        conversation_state=st.session_state,
                    )
                    if st.session_state.get("memory_persistence_enabled") and st.session_state.get("memory_recovery_key"):
                        try:
                            save_memory(
                                st.session_state.memory_recovery_key,
                                st.session_state.business_memory,
                            )
                        except (OSError, ValueError, sqlite3.Error):
                            # Avoid exposing filesystem details in the chat UI.
                            st.session_state.memory_notice = "Memory could not be saved on this run."
                if assistant_message.get("error"):
                    st.error(assistant_message["content"])
                else:
                    st.markdown(assistant_message["content"])


if __name__ == "__main__":
    main()
