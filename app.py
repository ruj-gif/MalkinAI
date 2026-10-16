"""Streamlit chat interface for the existing Malkin AI RAG backend."""

from __future__ import annotations

from typing import Any, Callable, MutableMapping

from answer import AnswerError, answer_question

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
MICROPHONE_MESSAGE = "Voice input is coming soon. We are not recording audio yet."
PROFILE_DEFAULTS = {
    "profile_idea": "",
    "profile_location": "",
    "profile_budget": "",
    "profile_stage": "Exploring",
    "profile_language": "English",
}


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Set initial conversation and profile state without resetting active values."""
    state.setdefault("messages", [])
    for key, value in PROFILE_DEFAULTS.items():
        state.setdefault(key, value)


def clear_profile(state: MutableMapping[str, Any]) -> None:
    """Clear only optional profile fields, leaving conversation history intact."""
    for key, value in PROFILE_DEFAULTS.items():
        state[key] = value


def build_profile_context(profile: dict[str, Any]) -> str:
    """Format optional profile details as explicitly unverified user context."""
    fields = (
        ("Business idea", profile.get("idea")),
        ("State or Union Territory", profile.get("location")),
        ("Approximate starting budget (INR)", profile.get("budget")),
        ("Business stage", profile.get("stage")),
    )
    lines = [f"- {label}: {str(value).strip()}" for label, value in fields if value and str(value).strip()]
    if not lines:
        return ""
    return "\n".join(lines) + "\nThese details are user-provided context, not verified facts or official eligibility evidence."


def safe_error_message(error: Exception) -> str:
    """Map backend exceptions to useful UI errors without exposing internals."""
    message = str(error).casefold()
    if "groq_api_key" in message:
        return "The answer service is not configured yet. Check the project’s local .env setup."
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
) -> dict[str, Any]:
    """Append a user turn and its backend response to the current session history."""
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")
    history.append({"role": "user", "content": question})
    context = build_profile_context(profile)
    try:
        answer = backend(question, user_context=context or None)
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerError("The answer service returned an empty response.")
        assistant_message = {"role": "assistant", "content": answer.strip()}
    except Exception as exc:  # UI boundary: never surface stack traces or raw payloads.
        assistant_message = {
            "role": "assistant",
            "content": safe_error_message(exc),
            "error": True,
        }
    history.append(assistant_message)
    return assistant_message


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

    with st.sidebar:
        st.markdown('<div class="malkin-brand">🌺 Malkin AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="malkin-tagline">Your idea. Your business. Your next step.</div>', unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigate", ["Chat", "About Malkin AI"], label_visibility="collapsed")
        st.markdown("## Your business profile")
        st.caption("Optional context helps personalize the conversation. It is not verified scheme information.")
        st.text_input("Business idea", placeholder="e.g. Pickle making", key="profile_idea")
        st.text_input("State or Union Territory", placeholder="e.g. Gujarat", key="profile_location")
        st.text_input("Approximate starting budget in INR", placeholder="e.g. 100000", key="profile_budget")
        st.selectbox("Business stage", ["Exploring", "Planning", "Already operating"], key="profile_stage")
        st.selectbox("Preferred language", ["English", "Hindi", "Gujarati", "Other"], key="profile_language")
        st.caption("Language preference is saved for future multilingual support; responses currently remain in English.")
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
                    )
                if assistant_message.get("error"):
                    st.error(assistant_message["content"])
                else:
                    st.markdown(assistant_message["content"])


if __name__ == "__main__":
    main()
