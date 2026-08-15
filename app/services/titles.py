"""Model-generated conversation titles for the assistant flow."""

from langchain_core.messages import HumanMessage, SystemMessage

MAX_TITLE_CHARACTERS = 30

# Structured output (response_format) is not supported by every OpenAI-compatible provider
# (e.g. DeepSeek), so the title is produced by a strict plain-text prompt and cleaned here.
_TITLE_PROMPT = SystemMessage(
    "You create short conversation titles for a chat history. Generate a concise, informative "
    "title for a conversation that starts with the given user message. Write it in the same "
    "language as the message, at most 30 characters. Respond with ONLY the title text: no "
    "quotes, no markdown, no explanation, no trailing punctuation."
)


async def generate_conversation_title(model, first_message: str) -> tuple[str, dict | None]:
    """Asks the model for a short conversation title and cleans the raw reply.

    Returns ``(title, usage_metadata)``; ``usage_metadata`` is the raw provider usage dict
    (``input_tokens``/``output_tokens``) or ``None`` when the provider reported none.
    """
    if not first_message.strip():
        return "", None
    response = await model.ainvoke([_TITLE_PROMPT, HumanMessage(content=first_message)])
    content = response.content if isinstance(response.content, str) else ""
    title = content.strip().strip("\"'`“”‘’").strip()
    if not title:
        return "", getattr(response, "usage_metadata", None)
    title = title.splitlines()[0].strip().rstrip("。.!！?？…")
    return title[:MAX_TITLE_CHARACTERS], getattr(response, "usage_metadata", None)
