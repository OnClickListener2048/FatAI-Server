"""全测试共享的进程级临时目录与数据库。

SQLAlchemy 的 engine 是 app.db 模块导入时按 DATABASE_URL 创建一次的单例;
unittest 在同一进程内按模块名字母序导入所有测试模块, 各自设置 DATABASE_URL
会互相覆盖(只有先导入者生效), 各自删除临时目录则会破坏后续模块。

因此所有测试模块必须统一从这里取同一个数据库文件与上传目录,
清理工作推迟到进程退出(atexit)统一执行。
"""

import asyncio
import os
import tempfile
from pathlib import Path

_shared_directory = tempfile.TemporaryDirectory(prefix="fatai-tests-")
SHARED_DIR = Path(_shared_directory.name)

DATABASE_URL = f"sqlite+aiosqlite:///{(SHARED_DIR / 'test.db').as_posix()}"
UPLOAD_DIRECTORY = str(SHARED_DIR / "uploads")


def _cleanup() -> None:
    try:
        from app.db import engine

        asyncio.run(engine.dispose())
    except Exception:
        pass
    try:
        _shared_directory.cleanup()
    except Exception:
        pass


import atexit

atexit.register(_cleanup)
