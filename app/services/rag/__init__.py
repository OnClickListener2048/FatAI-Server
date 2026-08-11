"""RAG 检索增强生成: embedding / 分块 / 向量存储 / 索引 / 检索。

设计要点:
- EmbeddingService 直连 OpenAI 兼容 /embeddings 端点(本地 Ollama 或托管 BYOK),零框架依赖
- VectorStore 持有专用 sqlite3 连接操作 vec0 虚拟表(sqlite-vec 不可用时退化为
  BLOB + Python 余弦扫描,行为与 scripts/rag_demo.py 一致)
- 索引与检索全在服务端;任何 RAG 失败都不得阻断聊天流(降级为按时间倒序取记忆)
"""
