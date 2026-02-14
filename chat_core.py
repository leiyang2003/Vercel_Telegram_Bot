"""
Minimal chat core for Vercel webhook: Grok backend, memory from workspace, no TTS/skills/vector.
WORKSPACE_DIR is read from os.environ["CHATBOT_WORKSPACE"] at runtime (set per request).
"""
import json
import os
import re
from pathlib import Path

import requests

# Read at runtime so webhook can set env per request
def _workspace_dir() -> Path:
    p = os.environ.get("CHATBOT_WORKSPACE", "").strip()
    return Path(p) if p else Path("/tmp/nowhere")

XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GROK_CHAT_MODEL = os.environ.get("GROK_CHAT_MODEL", "grok-3")
MAX_TOKENS = 4096
DEFAULT_SYSTEM = "You are a helpful assistant."
MEMORY_MAX_CONTEXT_CHARS = int(os.environ.get("MEMORY_MAX_CONTEXT_CHARS", "4000"))
MEMORY_FLUSH_HISTORY_TURNS = int(os.environ.get("MEMORY_FLUSH_HISTORY_TURNS", "3"))
MEMORY_FLUSH_ESTIMATED_TOKENS = int(os.environ.get("MEMORY_FLUSH_ESTIMATED_TOKENS", "8000"))
MEMORY_EXTRACT_MAX_TURNS = 5
REMEMBER_TRIGGER_PHRASES = (
    "记住", "记一下", "记着", "别忘了",
    "remember this", "save to memory", "remember that", "don't forget",
)
MEMORY_EXTRACT_SYSTEM = (
    "You are a memory extractor. Given the conversation, output what to add to long-term memory. "
    "Use exactly these section headers (one per block):\n"
    "## MEMORY.md\n<content for long-term facts>\n\n"
    "## memory/YYYY-MM-DD.md\n<content for today's log>\n\n"
    "Output only these sections with content; if nothing to store, output exactly: NOTHING"
)

_MEMORY_FLUSH_CURSOR_TURNS = 0


def _get_chat_config(backend: str, xai_key: str = "") -> tuple[str, dict]:
    if (backend or "").strip().lower() == "grok":
        key = (xai_key or "").strip() or (os.environ.get("XAI_API_KEY") or "").strip()
        return XAI_CHAT_URL, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    return XAI_CHAT_URL, {"Content-Type": "application/json"}


def _extract_api_error_message(exc: requests.RequestException, default: str) -> str:
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None) if resp else None
    if resp is None:
        return "Grok 连接失败（未收到响应），请检查网络或 api.x.ai。"
    text = (getattr(resp, "text", None) or "").strip()
    if not text and getattr(resp, "content", None):
        try:
            text = (resp.content or b"").decode("utf-8", errors="replace").strip()
        except Exception:
            pass
    try:
        err = resp.json()
        if isinstance(err, dict):
            msg = err.get("error") or {}
            if isinstance(msg, dict):
                msg = msg.get("message") or msg.get("detail") or ""
            else:
                msg = str(msg) if msg else ""
            if not msg:
                msg = err.get("message") or err.get("detail") or ""
            if msg and status is not None:
                return f"[{status}] {msg}"
            if msg:
                return msg
    except Exception:
        pass
    if text and len(text) <= 500:
        return f"[{status}] {text}" if status is not None else text
    return default


def _ensure_workspace_memory_dir() -> Path:
    w = _workspace_dir()
    if w and str(w) != "/tmp/nowhere":
        w.mkdir(parents=True, exist_ok=True)
        (w / "memory").mkdir(parents=True, exist_ok=True)
    return w


def read_memory_file(path: Path) -> str:
    try:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return ""


def append_to_memory_file(path: Path, content: str, daily_prefix: bool = False) -> None:
    if not content or not content.strip():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if daily_prefix:
                from datetime import datetime
                f.write("\n\n--- " + datetime.now().strftime("%Y-%m-%d %H:%M") + " ---\n\n")
            f.write(content.strip() + "\n")
    except Exception:
        pass


def get_memory_block_for_context(max_chars: int | None = None) -> str:
    _ensure_workspace_memory_dir()
    w = _workspace_dir()
    if not w or str(w) == "/tmp/nowhere":
        return ""
    max_chars = max_chars if max_chars is not None else MEMORY_MAX_CONTEXT_CHARS
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parts = []
    mem_md = w / "MEMORY.md"
    if mem_md.exists():
        content = read_memory_file(mem_md)
        if content.strip():
            parts.append("### MEMORY.md (long-term)\n\n" + content.strip())
    for date_label in (today, yesterday):
        daily = w / "memory" / f"{date_label}.md"
        if daily.exists():
            content = read_memory_file(daily)
            if content.strip():
                parts.append(f"### memory/{date_label}.md\n\n" + content.strip())
    if not parts:
        return ""
    combined = "\n\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[-max_chars:]
    return "\n\n## Long-term memory\n\n" + combined


def get_memory_block_for_turn(user_message: str, use_vector: bool = False) -> str:
    return get_memory_block_for_context()


def get_skills_block(include_exec_instructions: bool = False) -> str:
    return ""


def _history_to_tuples(history: list) -> list:
    if not history:
        return []
    if isinstance(history[0], dict):
        pairs = []
        i = 0
        while i < len(history):
            role = history[i].get("role", "")
            content = history[i].get("content", "") or ""
            if role == "user":
                asst = ""
                if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                    asst = history[i + 1].get("content", "") or ""
                    i += 1
                pairs.append((content, asst))
            i += 1
        return pairs
    return list(history)


def _tuples_to_messages(history: list) -> list:
    out = []
    for user, assistant in history:
        if user:
            out.append({"role": "user", "content": user})
        if assistant:
            out.append({"role": "assistant", "content": assistant})
    return out


def build_messages(
    character_prompt: str,
    history: list,
    user_message: str,
    memory_block: str = "",
    skills_block: str = "",
) -> list:
    system = (character_prompt or "").strip() or DEFAULT_SYSTEM
    if memory_block:
        system = system + memory_block
    if skills_block:
        system = system + skills_block
    messages = [{"role": "system", "content": system}]
    tuples = _history_to_tuples(history)
    for user, assistant in tuples:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


def _append_conversation_log(character_prompt: str, user_msg: str, assistant_reply: str) -> None:
    pass


def _maybe_is_game_guide_query(message: str) -> bool:
    if not (message or "").strip():
        return False
    text = (message or "").strip()
    lower = text.lower()
    if "攻略" not in text:
        return False
    keywords = ("游戏", "通关", "过关", "boss", "打法", "build", "流派", "加点", "天赋", "开荒", "前期")
    return any(k in text or k in lower for k in keywords)


def _call_grok_for_game_guide(message: str) -> str:
    api_key = (os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY") or "").strip()
    if not api_key:
        return "当前未配置 GROK_API_KEY/XAI_API_KEY，暂时无法获取游戏攻略。"
    try:
        url = "https://api.x.ai/v1/chat/completions"
        payload = {
            "model": "grok-4",
            "messages": [
                {"role": "system", "content": "You are a game strategy assistant. Answer in Chinese. Summarize key strategy in ~100 chars, then list 3-5 guide links."},
                {"role": "user", "content": message},
            ],
            "max_tokens": 512,
            "temperature": 0.5,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=60)
        r.raise_for_status()
        content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return (content or "").strip() or "暂无攻略内容。"
    except Exception:
        return "获取攻略时出错，请稍后再试。"


def _user_wants_remember(user_message: str) -> bool:
    if not (user_message or user_message.strip()):
        return False
    lower = user_message.strip().lower()
    return any(p in lower or p in user_message for p in REMEMBER_TRIGGER_PHRASES)


def _should_flush_memory_for_history(history_tuples: list, cursor: int) -> bool:
    n = len(history_tuples)
    delta = history_tuples[cursor:]
    if len(delta) >= MEMORY_FLUSH_HISTORY_TURNS:
        return True
    est = sum(len(u) + len(a) for u, a in delta)
    return (est // 4) >= MEMORY_FLUSH_ESTIMATED_TOKENS


def run_memory_extract_and_append(user_message: str, assistant_reply: str, character_prompt: str = "", extra_turns: list | None = None) -> None:
    from datetime import datetime
    _ensure_workspace_memory_dir()
    w = _workspace_dir()
    if not w or str(w) == "/tmp/nowhere":
        return
    today = datetime.now().strftime("%Y-%m-%d")
    system = MEMORY_EXTRACT_SYSTEM.replace("YYYY-MM-DD", today) + f"\n\nToday's date is {today}. Use ONLY memory/{today}.md for the daily log header."
    dialogue = f"User: {user_message}\nAssistant: {assistant_reply}"
    if extra_turns:
        for u, a in extra_turns[-MEMORY_EXTRACT_MAX_TURNS:]:
            dialogue = f"User: {u}\nAssistant: {a}\n\n" + dialogue
    key = (os.environ.get("XAI_API_KEY") or "").strip()
    if not key:
        return
    try:
        r = requests.post(
            XAI_CHAT_URL,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            json={"model": GROK_CHAT_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": dialogue}], "max_tokens": 512, "temperature": 0.2, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except Exception:
        return
    content = content.strip()
    if not content or content.upper() == "NOTHING":
        return
    mem_md = w / "MEMORY.md"
    daily_md = w / "memory" / f"{today}.md"
    in_memory, in_daily = False, False
    buf_memory, buf_daily = [], []
    for line in content.split("\n"):
        if re.match(r"^\s*##\s+MEMORY\.md\s*$", line, re.IGNORECASE):
            in_memory, in_daily = True, False
            continue
        if re.match(r"^\s*##\s+memory/.*\.md\s*$", line, re.IGNORECASE):
            in_daily, in_memory = True, False
            continue
        if in_memory:
            buf_memory.append(line)
        elif in_daily:
            buf_daily.append(line)
    if buf_memory:
        append_to_memory_file(mem_md, "\n".join(buf_memory).strip())
    if buf_daily:
        append_to_memory_file(daily_md, "\n".join(buf_daily).strip(), daily_prefix=True)


def stream_chat(
    character_prompt: str,
    history: list,
    message: str,
    enable_tts: bool = False,
    tts_api_key: str = "",
    tts_backend: str = "dashscope",
    lan_tts_url: str = "",
    lan_tts_voice: str = "",
    lan_tts_instruction: str = "",
    temperature: float = 0.7,
    use_memory: bool = True,
    use_vector_search: bool = False,
    enable_skill_exec: bool = False,
    chat_backend: str = "grok",
    xai_api_key: str = "",
):
    global _MEMORY_FLUSH_CURSOR_TURNS
    if not (message or message.strip()):
        yield _tuples_to_messages(_history_to_tuples(history)), "", "", None
        return
    if _maybe_is_game_guide_query(message):
        tuples = _history_to_tuples(history)
        user_msg = (message or "").strip()
        guide = _call_grok_for_game_guide(user_msg)
        full = guide or "我暂时无法获取该游戏的攻略。"
        _append_conversation_log(character_prompt or "", user_msg, full)
        yield _tuples_to_messages(tuples + [[user_msg, full]]), "", "", None
        return
    tuples = _history_to_tuples(history)
    new_tuples = tuples + [[message.strip(), ""]]
    yield _tuples_to_messages(new_tuples), "", "", None
    memory_block = get_memory_block_for_turn(message.strip(), use_vector=use_vector_search) if use_memory else ""
    skills_block = get_skills_block(include_exec_instructions=enable_skill_exec)
    messages = build_messages(character_prompt, history, message.strip(), memory_block=memory_block, skills_block=skills_block)
    payload = {"messages": messages, "max_tokens": MAX_TOKENS, "temperature": temperature, "stream": True}
    url, headers = _get_chat_config(chat_backend or "grok", xai_api_key or "")
    if (chat_backend or "").strip().lower() == "grok":
        payload["model"] = GROK_CHAT_MODEL
    if not (headers.get("Authorization") or "").replace("Bearer ", "").strip():
        yield _tuples_to_messages(tuples + [[message.strip(), "Grok AI 需要设置 XAI_API_KEY。"]]), "", "", None
        return
    try:
        resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        err_msg = _extract_api_error_message(e, "Grok 请求失败。")
        yield _tuples_to_messages(tuples + [[message.strip(), err_msg]]), "", "", None
        return
    full = ""
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload_line = line[6:].decode()
        if payload_line.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload_line)
            part = (chunk.get("choices", [{}])[0].get("delta", {}).get("content") or "")
            full += part
            yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", "", None
        except json.JSONDecodeError:
            continue
    _append_conversation_log(character_prompt or "", message.strip(), full)
    new_tuples = tuples + [[message.strip(), full]]
    wants = _user_wants_remember(message.strip())
    cursor = _MEMORY_FLUSH_CURSOR_TURNS
    if len(new_tuples) < cursor:
        _MEMORY_FLUSH_CURSOR_TURNS = 0
        cursor = 0
    should = _should_flush_memory_for_history(new_tuples, cursor)
    if wants or should:
        run_memory_extract_and_append(message.strip(), full, character_prompt or "", extra_turns=tuples)
        _MEMORY_FLUSH_CURSOR_TURNS = len(new_tuples)
    yield _tuples_to_messages(new_tuples), "", "", None
