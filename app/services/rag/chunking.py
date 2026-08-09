"""结构感知 Markdown 分块(从 scripts/chunker.py 学习稿正式化)。

策略(与学习稿一致):
    1. 按标题(#/##/###…)切分文档为 section,维护标题栈
    2. 标题路径拼到 chunk 内容前,让 embedding 知道文本属于哪个主题
    3. 超长 section 按句子边界做滑窗 + 重叠,防止语义被硬切
    4. 过短 section 与相邻同路径 section 合并,避免碎块
"""

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def _sliding_window(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """句子滑窗: 累积句子直到超限, 保留尾部 overlap 字符作为下块开头。"""
    if len(text) <= max_chars:
        return [text]
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current)
            tail = current[-overlap_chars:]
            boundary = max(tail.rfind("。"), tail.rfind("！"), tail.rfind("？"))
            current = tail[boundary + 1:] if boundary >= 0 else tail
        current += sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(md_text: str, max_chars: int = 800, overlap_chars: int = 120) -> list[dict]:
    """返回 [{path, content}]。path 是标题路径(如 "记忆系统 > 记忆提取")。"""
    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []
    current_body: list[str] = []

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if body:
            sections.append(([title for _, title in heading_stack], body))
        current_body.clear()

    for line in md_text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level, title = len(match.group(1)), match.group(2).strip()
            heading_stack = [item for item in heading_stack if item[0] < level]
            heading_stack.append((level, title))
        else:
            current_body.append(line)
    flush()

    raw_chunks: list[dict] = []
    for path, body in sections:
        prefix = " > ".join(path)
        for piece in _sliding_window(body, max_chars, overlap_chars):
            raw_chunks.append({"path": prefix, "content": f"{prefix}\n{piece}" if prefix else piece})

    merged: list[dict] = []
    for chunk in raw_chunks:
        if merged and len(chunk["content"]) < 60 and chunk["path"] == merged[-1]["path"]:
            merged[-1]["content"] += "\n" + chunk["content"]
        else:
            merged.append(dict(chunk))
    return merged


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    """纯文本分块(记忆条目等无标题结构): 句子滑窗 + 重叠。"""
    return _sliding_window(text.strip(), max_chars, overlap_chars)
