"""语义分块 + 结构感知(当前主流 RAG 分块方案)。

策略:
    1. 按标题切 section,标题路径拼到 chunk 前 —— 让 embedding 知道文本属于哪个主题
    2. 超长 section 内做语义分块: 批量嵌入句子,相邻句子相似度低的边界断开
       (LlamaIndex SemanticSplitterNodeParser 的思路); 相似度高于阈值的边界
       不硬切,保证语义连续的文本不被切断
    3. 目标块大小约为 max_chars 的 70%,允许块略超上限而不是在语义中途截断
    4. 过短 chunk 与相邻同 path chunk 合并,避免碎块

句子级嵌入需要一次额外的批量调用(本地 bge-m3 成本可接受);短文本直接
整块返回,不触发嵌入。
"""

import math
import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")

#: 相邻句子相似度高于此值则不断开(语义连续); bge-m3 句子对实测相关/无关
#: 大致在 0.6-0.7 与 0.4-0.5 之间,取 0.5 作为经验分界。
BREAK_THRESHOLD = 0.5
#: 语义目标块大小 = max_chars * TARGET_FRACTION
TARGET_FRACTION = 0.7


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def _extract_sections(md_text: str) -> list[tuple[list[str], str]]:
    """标题栈解析,返回 [(标题路径, 正文)]。"""
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
    return sections


def _cosine(a: list[float], b: list[float]) -> float:
    """向量已由 EmbeddingService 归一化,点积即余弦相似度。"""
    return sum(x * y for x, y in zip(a, b))


def semantic_breakpoints(
    similarities: list[float],
    sentence_chars: list[int],
    max_chars: int,
    target_chars: int,
    threshold: float = BREAK_THRESHOLD,
) -> set[int]:
    """在句子边界上选择断点。

    similarities[i] 是第 i 与第 i+1 句的相似度; 返回值中的位置 i 表示在
    i 与 i+1 之间断开。规则:
        - 目标断点数由总字符/目标块大小决定
        - 相似度最低的边界优先断开
        - 相似度高于 threshold 的边界绝不硬断(语义连续)
    """
    n = len(sentence_chars)
    if n <= 1 or not similarities:
        return set()
    total_chars = sum(sentence_chars)
    target_breaks = max(1, math.ceil(total_chars / target_chars)) - 1
    order = sorted(range(n - 1), key=lambda i: similarities[i])
    breaks: set[int] = set()
    for boundary in order:
        if len(breaks) >= target_breaks:
            break
        if similarities[boundary] > threshold:
            break
        breaks.add(boundary)
    return breaks


async def _semantic_split(body: str, embedder, max_chars: int) -> list[str]:
    """对单个 section 正文做语义分块。短文本直接整块返回。"""
    if len(body) <= max_chars:
        return [body]
    sentences = _split_sentences(body)
    if len(sentences) <= 1:
        return [body]
    vectors = await embedder.embed(sentences)
    similarities = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    sentence_chars = [len(sentence) for sentence in sentences]
    breaks = semantic_breakpoints(
        similarities,
        sentence_chars,
        max_chars,
        int(max_chars * TARGET_FRACTION),
    )
    if not breaks and len(body) > max_chars * 2:
        # 语义上无处断开(段落内容高度连续)但文本过长: 句子滑窗兜底,
        # 避免产出远超 max_chars 的单块
        chunks, current = [], ""
        for sentence in sentences:
            if len(current) + len(sentence) > max_chars and current:
                chunks.append(current)
                current = ""
            current += sentence
        if current:
            chunks.append(current)
        return chunks
    chunks: list[str] = []
    current: list[str] = []
    for index, sentence in enumerate(sentences):
        current.append(sentence)
        if index in breaks:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return chunks


async def chunk_markdown(
    md_text: str,
    embedder,
    max_chars: int = 800,
) -> list[dict]:
    """返回 [{path, content}]。path 是标题路径(如 "记忆系统 > 记忆提取")。"""
    raw_chunks: list[dict] = []
    for path, body in _extract_sections(md_text):
        prefix = " > ".join(path)
        for piece in await _semantic_split(body, embedder, max_chars):
            raw_chunks.append({"path": prefix, "content": f"{prefix}\n{piece}" if prefix else piece})

    merged: list[dict] = []
    for chunk in raw_chunks:
        if merged and len(chunk["content"]) < 60 and chunk["path"] == merged[-1]["path"]:
            merged[-1]["content"] += "\n" + chunk["content"]
        else:
            merged.append(dict(chunk))
    return merged


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """纯文本分块(记忆条目等无标题结构)。

    记忆内容有 400 字符上限、远小于分块上限,通常整块返回; 超长时在句子
    边界硬切(语义分块对短记忆没有收益)。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current)
            current = ""
        current += sentence
    if current:
        chunks.append(current)
    return chunks
