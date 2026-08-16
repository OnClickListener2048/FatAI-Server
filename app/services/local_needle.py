"""Local conversation titles from the on-device needle2 engine (Windows-only).

Needle2 (Cactus-Compute/needle2) is a small tool-calling model: it never emits free text,
every turn is JSON with ``function_calls``. The ``cactus-needle`` wheel embeds the engine
and the model in ``libneedle.dll``, so no weights file is needed. The title task is declared
as the single ``set_title`` tool, the first user message goes into the input text (needle's
system slot only carries environment facts), and ``function_calls[0].arguments.title`` is
the answer. An empty ``function_calls`` list is the model's refusal — the caller then falls
back to the cloud.

The package is Windows-only and not on PyPI; it is installed manually into the venv (see
CLAUDE.md). The import is lazy and every failure returns ``None``, so servers without the
wheel simply skip local generation. The engine call blocks (~1s), so callers must invoke
this off the event loop via ``asyncio.to_thread``.
"""

import logging
from datetime import date

from app.services.titles import MAX_TITLE_CHARACTERS, clean_title

logger = logging.getLogger(__name__)

_SET_TITLE_SCHEMA = {
    "name": "set_title",
    "description": "Create a concise title for a chat conversation based on its first user message.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": (
                    "The conversation title, plain text, in the same language as the message, "
                    f"at most {MAX_TITLE_CHARACTERS} characters, no quotes or markdown."
                ),
            }
        },
        "required": ["title"],
    },
}

# The 45M model echoes the first user message as the title (capped by MAX_TITLE_CHARACTERS
# in clean_title), which is the intended quality ceiling for the local path. Instruction
# must precede the message: with the message first the model refuses. Keep the wording
# minimal — longer phrasing makes it echo instruction words instead of the message.
_INSTRUCTION = (
    "Create a concise title for a chat conversation that starts with this first user "
    "message. Call set_title with the title."
)


def _system_facts() -> str:
    return f"date: {date.today().isoformat()}, locale: zh-CN"


def generate_title_local(first_message: str) -> str | None:
    """Titles a conversation with the local needle2 engine.

    Returns the cleaned title, or ``None`` when needle is unavailable (wheel not installed),
    refused the request, or failed — the caller falls back to the cloud then. Every turn
    starts a fresh session, so consecutive titles never pollute each other's context.
    """
    if not first_message.strip():
        return None
    try:
        from needle import Needle
    except ImportError:
        logger.debug("cactus-needle not installed; skipping local title generation")
        return None
    try:
        agent = Needle(tools=[_SET_TITLE_SCHEMA], system=_system_facts())
        response = agent.complete(f"{_INSTRUCTION}\n\n{first_message}", 256)
    except Exception:
        logger.warning("needle title generation failed", exc_info=True)
        return None
    for call in response.get("function_calls") or []:
        if call.get("name") == "set_title":
            title = (call.get("arguments") or {}).get("title")
            if isinstance(title, str):
                return clean_title(title)
            break
    return None
