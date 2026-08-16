"""Model-generated conversation titles for the assistant flow."""

from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

MAX_TITLE_CHARACTERS = 20

# How many opening turns are fed to the title model. The first user message alone is often
# a greeting or an attachment mention, so both sides of the opening exchange give the model
# enough context to summarize the actual question.
MAX_TITLE_TRANSCRIPT_MESSAGES = 4

# Structured output (response_format) is not supported by every OpenAI-compatible provider
# (e.g. DeepSeek), so the title is produced by a strict plain-text prompt and cleaned here.
_TITLE_PROMPT = SystemMessage(
    f"You create short conversation titles for a chat history. The conversation opens with "
    f"a user question and an assistant answer. Summarize the core question or topic of the "
    f"conversation in a concise, informative title. Write it in the same language as the "
    f"conversation, at most {MAX_TITLE_CHARACTERS} characters. Respond with ONLY the title "
    f"text: no quotes, no markdown, no explanation, no trailing punctuation."
)


def transcript_for_title(
    messages: Sequence[tuple[str, str]], limit: int = MAX_TITLE_TRANSCRIPT_MESSAGES
) -> str:
    """Renders the first turns as ``User:/Assistant:`` lines for the title prompt.

    Drops roles other than user/assistant and empty content, keeping the title model's
    context tiny while still giving it the opening exchange to summarize the question.
    """
    lines: list[str] = []
    for role, content in messages[:limit]:
        label = "User" if role == "user" else "Assistant" if role == "assistant" else None
        content = (content or "").strip()
        if label is not None and content:
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


def clean_title(content: str) -> str:
    """Turns a raw model reply into a title: strips quotes/markdown, keeps the first line,
    drops trailing punctuation, and truncates to [MAX_TITLE_CHARACTERS]."""
    text = content.strip().strip("\"'`“”‘’").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip().rstrip("。.!！?？…")[:MAX_TITLE_CHARACTERS]


async def generate_conversation_title(model, transcript: str) -> tuple[str, dict | None]:
    """Asks the model for a short conversation title and cleans the raw reply.

    ``transcript`` is the opening exchange rendered by [transcript_for_title]; the model
    summarizes its core question into the title. Returns ``(title, usage_metadata)``;
    ``usage_metadata`` is the raw provider usage dict (``input_tokens``/``output_tokens``)
    or ``None`` when the provider reported none.
    """
    if not transcript.strip():
        return "", None
    response = await model.ainvoke([_TITLE_PROMPT, HumanMessage(content=transcript)])
    content = response.content if isinstance(response.content, str) else ""
    return clean_title(content), getattr(response, "usage_metadata", None)
