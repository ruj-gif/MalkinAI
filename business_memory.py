"""Small, explicit business-memory helpers for Malkin AI conversations.

Only details stated by the user are promoted into user memory. Conversation
history, assistant suggestions, and cited document facts stay in separate
sections and are never treated as eligibility evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any


MAX_RECENT_MESSAGES = 10
MAX_SUMMARY_CHARACTERS = 1800
INDIAN_STATES_AND_UTS = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
)


def empty_business_memory() -> dict[str, Any]:
    """Return a versioned schema containing user-provided facts only."""
    return {
        "version": 1,
        "user_provided": {
            "business_idea": "",
            "products": [],
            "state_or_location": "",
            "starting_budget": {"amount": "", "currency": ""},
            "business_stage": "",
            "target_customers": [],
            "goals_and_preferences": [],
            "decisions": [],
            "next_steps": [],
            "unresolved_questions": [],
        },
        "assistant_suggestions": [],
        "document_facts": [],
    }


def normalize_business_memory(value: Any) -> dict[str, Any]:
    """Validate memory and discard unknown fields or malformed values safely."""
    clean = empty_business_memory()
    if not isinstance(value, dict):
        return clean
    facts = value.get("user_provided")
    if not isinstance(facts, dict):
        return clean

    for key in ("business_idea", "state_or_location", "business_stage"):
        item = facts.get(key)
        if isinstance(item, str):
            clean["user_provided"][key] = _clean_text(item)

    products = facts.get("products")
    if isinstance(products, list):
        clean["user_provided"]["products"] = _clean_list(products)

    budget = facts.get("starting_budget")
    if isinstance(budget, dict):
        amount = budget.get("amount")
        currency = budget.get("currency")
        if isinstance(amount, (str, int, float)) and not isinstance(amount, bool):
            clean["user_provided"]["starting_budget"]["amount"] = _clean_text(str(amount))
        if isinstance(currency, str):
            clean["user_provided"]["starting_budget"]["currency"] = _clean_text(currency).upper()[:12]

    for key in (
        "target_customers", "goals_and_preferences", "decisions", "next_steps",
        "unresolved_questions",
    ):
        if isinstance(facts.get(key), list):
            clean["user_provided"][key] = _clean_list(facts[key])

    # These sections are purposefully not populated from conversation text.
    # They remain separate so future features cannot mistake suggestions or
    # document-derived facts for user-confirmed business details.
    clean["assistant_suggestions"] = _clean_list(value.get("assistant_suggestions", []))
    clean["document_facts"] = _clean_list(value.get("document_facts", []))
    return clean


def _clean_text(value: str) -> str:
    return " ".join(value.split())[:500]


def _clean_list(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            item = _clean_text(value)
            if item and item not in result:
                result.append(item)
        if len(result) >= 20:
            break
    return result


def _replace_once(values: list[str], value: str) -> list[str]:
    clean_value = _clean_text(value)
    return [clean_value] if clean_value else values


def update_memory_from_user(memory: Any, message: str) -> dict[str, Any]:
    """Conservatively capture explicit facts from the latest user message."""
    clean = normalize_business_memory(memory)
    if not isinstance(message, str) or not message.strip():
        return clean
    text = " ".join(message.split())
    facts = clean["user_provided"]

    match = re.search(
        r"\b(?:i want to (?:start|launch|open|build)|my business idea is|my business is)\s+"
        r"(?:an?\s+)?([^,.!?;]+)", text, re.IGNORECASE,
    )
    if match:
        idea = match.group(1)
        for state_name in INDIAN_STATES_AND_UTS:
            idea = re.sub(rf"\s+in\s+{re.escape(state_name)}$", "", idea, flags=re.IGNORECASE)
        facts["business_idea"] = _clean_text(idea)

    match = re.search(r"\b(?:i make|i sell|i produce|my products are)\s+([^.!?;]+)", text, re.IGNORECASE)
    if match:
        facts["products"] = _replace_once(facts["products"], match.group(1))

    match = re.search(r"\b(?:my customers are|i sell to|i want to sell to)\s+([^.!?;]+)", text, re.IGNORECASE)
    if match:
        facts["target_customers"] = _replace_once(facts["target_customers"], match.group(1))

    state_regex = "|".join(re.escape(name) for name in sorted(INDIAN_STATES_AND_UTS, key=len, reverse=True))
    match = re.search(rf"\b(?:my state is|state is|based in|located in|operating in|moved to|in)\s+({state_regex})\b", text, re.IGNORECASE)
    if match:
        facts["state_or_location"] = _clean_text(match.group(1))

    match = re.search(
        r"\b(?:my (?:starting )?budget is|budget is|i have|i can invest|starting with)\s+"
        r"(₹|\$|€|£|rs\.?|inr|usd|eur|gbp)?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(inr|usd|eur|gbp|rupees?|dollars?|euros?|pounds?)?\b",
        text, re.IGNORECASE,
    )
    if match:
        currency_token = (match.group(1) or match.group(3) or "").casefold()
        currency_map = {
            "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
            "$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD",
            "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
            "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
        }
        facts["starting_budget"] = {
            "amount": match.group(2),
            "currency": currency_map.get(currency_token, ""),
        }

    stage_patterns = (
        (r"\b(?:already operating|already running|currently running)\b", "Already operating"),
        (r"\b(?:planning to start|in the planning stage|i am planning)\b", "Planning"),
        (r"\b(?:just exploring|still exploring|at the idea stage)\b", "Exploring"),
    )
    for pattern, stage in stage_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            facts["business_stage"] = stage
            break

    explicit_patterns = (
        ("goals_and_preferences", r"\b(?:my goal is|i prefer|i would prefer)\s+([^.!?;]+)"),
        ("decisions", r"\b(?:i decided to|i have decided to)\s+([^.!?;]+)"),
        ("next_steps", r"\b(?:i plan to|my next step is)\s+([^.!?;]+)"),
        ("unresolved_questions", r"\b(?:i still need to know|i am unsure about|i'm unsure about)\s+([^.!?;]+)"),
    )
    for key, pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            facts[key] = _replace_once(facts[key], match.group(1))
    return normalize_business_memory(clean)


def record_assistant_suggestions(memory: Any, answer: str) -> dict[str, Any]:
    """Keep rendered general suggestions separately from user-confirmed facts."""
    clean = normalize_business_memory(memory)
    if not isinstance(answer, str):
        return clean
    capture = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.casefold() == "general suggestions":
            capture = True
            continue
        if capture and stripped and not stripped.startswith("-"):
            capture = False
        if capture and stripped.startswith("-"):
            suggestion = _clean_text(stripped[1:].strip())
            if suggestion and suggestion not in clean["assistant_suggestions"]:
                clean["assistant_suggestions"].append(suggestion)
    clean["assistant_suggestions"] = clean["assistant_suggestions"][-10:]
    return normalize_business_memory(clean)


def merge_profile_into_memory(memory: Any, profile: dict[str, Any]) -> dict[str, Any]:
    """Merge non-empty sidebar facts; new values replace older ones."""
    clean = normalize_business_memory(memory)
    facts = clean["user_provided"]
    for profile_key, memory_key in (
        ("idea", "business_idea"),
        ("location", "state_or_location"),
        ("budget", "starting_budget"),
        ("stage", "business_stage"),
    ):
        value = profile.get(profile_key)
        if isinstance(value, str) and value.strip():
            if profile_key == "budget":
                facts[memory_key] = {"amount": _clean_text(value), "currency": "INR"}
            elif profile_key == "stage" and value == "Select a stage":
                # The placeholder is not an explicit user statement.
                continue
            else:
                facts[memory_key] = _clean_text(value)
    return normalize_business_memory(clean)


def build_memory_context(memory: Any) -> str:
    """Serialize only user-confirmed facts as unverified personalization context."""
    clean = normalize_business_memory(memory)
    facts = clean["user_provided"]
    populated = {
        key: value for key, value in facts.items()
        if value and value != {"amount": "", "currency": ""}
    }
    if not populated:
        return ""
    return json.dumps({"origin": "user_provided_unverified", "facts": populated}, ensure_ascii=False)


def build_retrieval_query(
    question: str,
    history: list[dict[str, Any]] | None = None,
    memory: Any = None,
) -> str:
    """Resolve short references with recent topic context for the existing retriever."""
    parts = [question.strip()]
    for message in (history or [])[-4:]:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip()[:500])
    facts = normalize_business_memory(memory)["user_provided"]
    parts.extend(value for value in (facts["business_idea"], *facts["products"]) if value)
    return "\n".join(dict.fromkeys(parts))[:5000]


def roll_conversation_history(
    summary: str,
    history: list[dict[str, Any]],
    max_recent_messages: int = MAX_RECENT_MESSAGES,
) -> tuple[str, list[dict[str, Any]]]:
    """Move old messages into a compact, clearly untrusted rolling transcript."""
    recent_limit = max(2, max_recent_messages - max_recent_messages % 2)
    if len(history) <= recent_limit:
        return summary, list(history)
    move_count = len(history) - recent_limit
    move_count += move_count % 2
    older, recent = history[:move_count], history[move_count:]
    lines = [summary.strip()] if summary.strip() else []
    for message in older:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            lines.append(f"{message['role']}: {' '.join(content.split())[:220]}")
    new_summary = "\n".join(lines)[-MAX_SUMMARY_CHARACTERS:]
    return new_summary, recent


def memory_to_json(memory: Any) -> str:
    return json.dumps(normalize_business_memory(memory), ensure_ascii=False, indent=2)


def memory_from_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        raise ValueError("Business memory must be valid JSON.") from None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("user_provided"), dict):
        raise ValueError("Business memory JSON has an unsupported structure.")
    return normalize_business_memory(parsed)
