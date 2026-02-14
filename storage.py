"""
存储抽象层：支持本地文件系统与 Vercel Blob。
当 BLOB_READ_WRITE_TOKEN 存在时使用 Blob，否则使用 users/ 目录。
"""
import json
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
BLOB_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()
_use_blob = bool(BLOB_TOKEN)


def _blob_path(*parts: str) -> str:
    return "/".join(parts).replace("//", "/").lstrip("/")


def _read_blob(path: str) -> Optional[bytes]:
    """从 Blob 读取内容，不存在返回 None"""
    if not _use_blob:
        return None
    try:
        import vercel_blob
        import requests
        out = vercel_blob.list({"limit": "1000"})
        blobs = out.get("blobs", []) if isinstance(out, dict) else []
        for b in blobs:
            p = b.get("pathname", "")
            if p == path or p.lstrip("/") == path.lstrip("/"):
                url = b.get("url") or b.get("downloadUrl")
                if url:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    return r.content
        return None
    except Exception:
        return None


def _write_blob(path: str, data: bytes) -> bool:
    """写入 Blob。成功返回 True，失败抛出异常（带原始错误信息）"""
    if not _use_blob:
        return False
    try:
        import vercel_blob
        vercel_blob.put(path, data)
        return True
    except Exception as e:
        raise RuntimeError(f"Blob write failed for {path}: {e}") from e


def _read_local(path: Path) -> Optional[bytes]:
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _write_local(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def storage_read(user_id: str, rel_path: str) -> Optional[bytes]:
    """读取用户文件。rel_path 如 'telegram_bots_config.json' 或 'personas/bot_1/persona.txt'"""
    blob_key = _blob_path("users", user_id, rel_path)
    if _use_blob:
        data = _read_blob(blob_key)
        if data is not None:
            return data
    local = ROOT / "users" / user_id / rel_path
    return _read_local(local)


def storage_write(user_id: str, rel_path: str, data: bytes) -> None:
    """写入用户文件"""
    blob_key = _blob_path("users", user_id, rel_path)
    if _use_blob:
        if _write_blob(blob_key, data):
            return
        # Blob 写入失败会抛异常，不会走到这里
    local = ROOT / "users" / user_id / rel_path
    try:
        _write_local(local, data)
    except OSError as e:
        raise RuntimeError(f"Local write failed for {rel_path}: {e}") from e


def storage_read_text(user_id: str, rel_path: str, default: str = "") -> str:
    data = storage_read(user_id, rel_path)
    if data is None:
        return default
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return default


def storage_write_text(user_id: str, rel_path: str, text: str) -> None:
    storage_write(user_id, rel_path, text.encode("utf-8"))


def storage_read_json(user_id: str, rel_path: str, default=None):
    data = storage_read(user_id, rel_path)
    if data is None:
        return default if default is not None else {}
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default if default is not None else {}


def storage_write_json(user_id: str, rel_path: str, obj: dict) -> None:
    storage_write(user_id, rel_path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def storage_path_exists(user_id: str, rel_path: str) -> bool:
    """检查文件是否存在"""
    if _use_blob:
        return storage_read(user_id, rel_path) is not None
    return (ROOT / "users" / user_id / rel_path).exists()


def get_user_dir(user_id: str) -> Path:
    """返回本地用户目录（用于 export 时写入 chat_logs 等，或本地模式）"""
    return ROOT / "users" / user_id


def ensure_user_dirs(user_id: str, slot_id: str) -> None:
    """确保用户 workspace/personas 目录存在（仅本地模式；Blob 模式无需创建）"""
    if _use_blob:
        return
    ud = get_user_dir(user_id)
    (ud / "workspace" / slot_id / "memory").mkdir(parents=True, exist_ok=True)
    mem_md = ud / "workspace" / slot_id / "MEMORY.md"
    if not mem_md.exists():
        mem_md.write_text("", encoding="utf-8")
    (ud / "personas" / slot_id).mkdir(parents=True, exist_ok=True)
