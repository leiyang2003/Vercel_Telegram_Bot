"""
QX Telegram bot — ChatBot_OpenClaw (memory, skills, game guide) + Visual Snapshot (/snap), Grok backend.
Requires: TELEGRAM_BOT_TOKEN, XAI_API_KEY in .env
Optional: TELEGRAM_PERSONA_FILE=Ani.txt, CHATBOT_WORKSPACE, SKILL_EXEC_ENABLED, MEMORY_SEARCH_ENABLED, OPENAI_API_KEY
"""

import asyncio
import html as html_module
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))

from dotenv import load_dotenv
load_dotenv(_script_dir / ".env")

# Load character persona from TELEGRAM_PERSONA_FILE (path relative to script dir)
_persona_filename = os.environ.get("TELEGRAM_PERSONA_FILE", "").strip()
_persona_path = None
if _persona_filename:
    _persona_path = (_script_dir / _persona_filename).resolve()
    if not _persona_path.exists():
        _persona_path = (_script_dir / "personas" / _persona_filename).resolve()
    if _persona_path.exists():
        try:
            DEFAULT_PERSONA = _persona_path.read_text(encoding="utf-8").strip()
        except Exception:
            DEFAULT_PERSONA = "You are a helpful assistant."
    else:
        DEFAULT_PERSONA = "You are a helpful assistant."
else:
    DEFAULT_PERSONA = "You are a helpful assistant."

# Import ChatBot_OpenClaw (memory, skills, game guide, conversation log)
import ChatBot_OpenClaw as _cb
stream_chat = _cb.stream_chat
_append_conversation_log = _cb._append_conversation_log
_history_to_tuples = _cb._history_to_tuples

# Import run_snap from dynamic-prompt (Visual Snapshot)
import importlib.util
_spec = importlib.util.spec_from_file_location("dynamic_prompt", _script_dir / "dynamic-prompt.py")
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)
run_snap = _dp.run_snap
CHAT_LOG_DIR = _dp.CHAT_LOG_DIR

# Per-bot state file so each bot has its own chat state (fixes shared Ani persona when switching bots)
_workspace_env = (os.environ.get("CHATBOT_WORKSPACE") or "").strip()
if _workspace_env:
    _bot_id = Path(_workspace_env).name
    TELEGRAM_STATES_FILE = CHAT_LOG_DIR / _bot_id / "qx_states.json"
    TELEGRAM_STATES_FILE.parent.mkdir(parents=True, exist_ok=True)
else:
    TELEGRAM_STATES_FILE = CHAT_LOG_DIR / "qx_states.json"

# --- Configuration HTML (mask secrets: show only set/not set or last 4 chars) ---
_SECRET_KEYS = frozenset({"TELEGRAM_BOT_TOKEN", "XAI_API_KEY", "OPENAI_API_KEY", "GROK_API_KEY", "DASHSCOPE_API_KEY"})


def _mask_env_value(key: str, value: str) -> str:
    if not value:
        return "(not set)"
    if key in _SECRET_KEYS:
        return "***" + value[-4:] if len(value) >= 4 else "***"
    return value


def _build_config() -> dict:
    """Build config dict for QX: env (masked), paths, persona, memory files, snap rules."""
    env_vars = [
        "TELEGRAM_PERSONA_FILE", "TELEGRAM_BOT_TOKEN", "XAI_API_KEY", "CHATBOT_WORKSPACE", "SNAP_NUM_IMAGES",
        "SKILL_EXEC_ENABLED", "SKILL_EXEC_TIMEOUT", "MEMORY_SEARCH_ENABLED", "OPENAI_API_KEY",
        "GROK_CHAT_MODEL", "GROK_API_KEY", "DASHSCOPE_API_KEY",
        "MEMORY_MAX_CONTEXT_CHARS", "MEMORY_FLUSH_HISTORY_TURNS", "MEMORY_FLUSH_ESTIMATED_TOKENS",
        "MEMORY_INDEX_PATH", "MEMORY_EMBEDDING_PROVIDER", "MEMORY_MAX_RESULTS", "MEMORY_MAX_SNIPPET_CHARS",
    ]
    env = {k: _mask_env_value(k, (os.environ.get(k) or "").strip()) for k in env_vars}

    workspace_dir = getattr(_cb, "WORKSPACE_DIR", Path("."))
    mem_md = workspace_dir / "MEMORY.md"
    memory_dir = workspace_dir / "memory"
    paths = {
        "persona_file": str(_persona_path) if _persona_path else "(none; using default)",
        "persona_file_exists": _persona_path.exists() if _persona_path else False,
        "TELEGRAM_STATES_FILE": str(TELEGRAM_STATES_FILE),
        "CHAT_LOG_DIR": str(CHAT_LOG_DIR),
        "WORKSPACE_DIR": str(workspace_dir),
        "MEMORY.md": str(mem_md),
        "memory_dir": str(memory_dir),
        "MEMORY_INDEX_PATH": str(getattr(_cb, "MEMORY_INDEX_PATH", "")),
        "CONVERSATIONS_LOG": str(getattr(_cb, "CONVERSATIONS_LOG", "")),
        "SNAP_LOG_DIR": str(getattr(_dp, "SNAP_LOG_DIR", "")),
        "VSNAPSHOT_DIR": str(getattr(_dp, "VSNAPSHOT_DIR", "")),
        "SNAP_CHARACTER_DESIGN_FILE": str(getattr(_dp, "SNAP_CHARACTER_DESIGN_FILE", "")),
        "story.md": str(CHAT_LOG_DIR / "story.md"),
    }

    persona_preview = (DEFAULT_PERSONA or "").strip()
    if len(persona_preview) > 8000:
        persona_preview = persona_preview[:8000] + "\n\n… (truncated)"

    memory_files = []
    if mem_md.exists():
        try:
            c = mem_md.read_text(encoding="utf-8", errors="replace").strip()
            memory_files.append({"path": str(mem_md), "size": mem_md.stat().st_size, "preview": c[:2000] + ("…" if len(c) > 2000 else "")})
        except Exception:
            memory_files.append({"path": str(mem_md), "size": 0, "preview": "(read error)"})
    else:
        memory_files.append({"path": str(mem_md), "size": 0, "preview": "(file not found)"})
    if memory_dir.is_dir():
        for f in sorted(memory_dir.glob("*.md")):
            try:
                c = f.read_text(encoding="utf-8", errors="replace").strip()
                memory_files.append({"path": str(f), "size": f.stat().st_size, "preview": c[:2000] + ("…" if len(c) > 2000 else "")})
            except Exception:
                memory_files.append({"path": str(f), "size": 0, "preview": "(read error)"})

    memory_rules = {
        "MEMORY_EXTRACT_SYSTEM": getattr(_cb, "MEMORY_EXTRACT_SYSTEM", ""),
        "REMEMBER_TRIGGER_PHRASES": getattr(_cb, "REMEMBER_TRIGGER_PHRASES", ()),
        "MEMORY_MAX_CONTEXT_CHARS": getattr(_cb, "MEMORY_MAX_CONTEXT_CHARS", ""),
        "MEMORY_FLUSH_HISTORY_TURNS": getattr(_cb, "MEMORY_FLUSH_HISTORY_TURNS", ""),
        "MEMORY_FLUSH_ESTIMATED_TOKENS": getattr(_cb, "MEMORY_FLUSH_ESTIMATED_TOKENS", ""),
    }

    snap_char_design = ""
    snap_char_path = getattr(_dp, "SNAP_CHARACTER_DESIGN_FILE", None)
    if snap_char_path and Path(snap_char_path).exists():
        try:
            snap_char_design = Path(snap_char_path).read_text(encoding="utf-8", errors="replace").strip()
            if len(snap_char_design) > 4000:
                snap_char_design = snap_char_design[:4000] + "\n\n… (truncated)"
        except Exception:
            snap_char_design = "(read error)"
    story_path = CHAT_LOG_DIR / "story.md"
    story_preview = ""
    if story_path.exists():
        try:
            story_content = story_path.read_text(encoding="utf-8", errors="replace").strip()
            story_preview = story_content[:2000] + ("…" if len(story_content) > 2000 else "")
        except Exception:
            story_preview = "(read error)"

    snap_rules = {
        "SNAP_NUM_IMAGES": getattr(_dp, "SNAP_NUM_IMAGES", ""),
        "SNAP_MAX_TURNS": getattr(_dp, "SNAP_MAX_TURNS", ""),
        "SNAP_SUMMARY_PROMPT": getattr(_dp, "SNAP_SUMMARY_PROMPT", ""),
        "SNAP_SCENE_SYSTEM": getattr(_dp, "SNAP_SCENE_SYSTEM", ""),
        "SNAP_SAFETY_REWRITE_PROMPT": getattr(_dp, "SNAP_SAFETY_REWRITE_PROMPT", ""),
        "SNAP_STORY_SYSTEM": getattr(_dp, "SNAP_STORY_SYSTEM", ""),
        "SNAP_THREE_ILLUST_SYSTEM_FOUR_LINES": getattr(_dp, "SNAP_THREE_ILLUST_SYSTEM_FOUR_LINES", ""),
        "SNAP_THREE_ILLUST_SYSTEM_THREE_LINES": getattr(_dp, "SNAP_THREE_ILLUST_SYSTEM_THREE_LINES", ""),
        "SNAP_IMAGE_STYLE": getattr(_dp, "SNAP_IMAGE_STYLE", ""),
        "XAI_IMAGINE_URL": getattr(_dp, "XAI_IMAGINE_URL", ""),
        "XAI_IMAGINE_MODEL": getattr(_dp, "XAI_IMAGINE_MODEL", ""),
        "SNAP_CHARACTER_DESIGN_FILE_content": snap_char_design or "(file not found or empty)",
        "story.md_preview": story_preview or "(file not found or empty)",
    }

    commands = {
        "/start": "Welcome message; lists /snap.",
        "/snap": f"Generate Visual Snapshot (story + {_dp.SNAP_NUM_IMAGES} images) from the last 2 turns + context.",
        "/clear": "Clear this chat's history and global long-term memory (MEMORY.md, memory/*.md, vector index).",
    }

    return {
        "env": env,
        "paths": paths,
        "commands": commands,
        "persona_preview": persona_preview,
        "memory_files": memory_files,
        "memory_rules": memory_rules,
        "snap_rules": snap_rules,
    }


def _render_config_html(config: dict) -> str:
    """Render config dict to a single HTML string."""
    def esc(s: str) -> str:
        return html_module.escape(str(s))

    lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>QX Configuration</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:1rem;max-width:900px;}",
        "h1{font-size:1.25rem;} h2{font-size:1rem;margin-top:1.25rem;}",
        "pre,code{background:#f0f0f0;padding:0.25rem 0.5rem;overflow-x:auto;font-size:0.875rem;}",
        "pre{white-space:pre-wrap;word-break:break-word;}</style></head><body>",
        "<h1>QX Configuration</h1>",
        "<p>Generated at startup. Re-run QX.py to refresh.</p>",
    ]

    lines.append("<h2>Environment</h2><table border='1' cellpadding='4' style='border-collapse:collapse;'>")
    for k, v in config["env"].items():
        lines.append(f"<tr><td><code>{esc(k)}</code></td><td><code>{esc(v)}</code></td></tr>")
    lines.append("</table>")

    lines.append("<h2>Paths</h2><table border='1' cellpadding='4' style='border-collapse:collapse;'>")
    for k, v in config["paths"].items():
        lines.append(f"<tr><td><code>{esc(k)}</code></td><td><code>{esc(v)}</code></td></tr>")
    lines.append("</table>")

    lines.append("<h2>Telegram commands</h2><table border='1' cellpadding='4' style='border-collapse:collapse;'>")
    for cmd, desc in config.get("commands", {}).items():
        lines.append(f"<tr><td><code>{esc(cmd)}</code></td><td>{esc(desc)}</td></tr>")
    lines.append("</table>")

    lines.append("<h2>System prompt (persona)</h2>")
    lines.append(f"<pre>{esc(config['persona_preview'])}</pre>")

    lines.append("<h2>Memory</h2>")
    lines.append("<h3>Memory rules</h3><table border='1' cellpadding='4' style='border-collapse:collapse;'>")
    for k, v in config["memory_rules"].items():
        disp = str(v)[:500] + ("…" if len(str(v)) > 500 else "")
        lines.append(f"<tr><td><code>{esc(k)}</code></td><td><pre>{esc(disp)}</pre></td></tr>")
    lines.append("</table>")
    lines.append("<h3>Memory files</h3>")
    for m in config["memory_files"]:
        lines.append(f"<p><strong>{esc(m['path'])}</strong> (size: {m['size']})</p><pre>{esc(m['preview'])}</pre>")

    lines.append("<h2>/snap rules</h2>")
    for k, v in config["snap_rules"].items():
        disp = str(v)[:2000] + ("…" if len(str(v)) > 2000 else "")
        lines.append(f"<p><strong>{esc(k)}</strong></p><pre>{esc(disp)}</pre>")

    lines.append("</body></html>")
    return "\n".join(lines)


def _run_config_browser():
    """Daemon: build config, write qx_config.html, open in browser."""
    try:
        config = _build_config()
        html = _render_config_html(config)
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CHAT_LOG_DIR / "qx_config.html"
        out_path.write_text(html, encoding="utf-8")
        time.sleep(0.3)
        webbrowser.open(out_path.as_uri())
    except Exception:
        pass


def _load_states() -> dict:
    """Load persisted chat states from JSON. Returns {chat_id: {history, character_prompt}} with int keys."""
    if not TELEGRAM_STATES_FILE.exists():
        return {}
    try:
        with open(TELEGRAM_STATES_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        out = {}
        for k, v in raw.items():
            if not isinstance(v, dict):
                continue
            history = v.get("history")
            if not isinstance(history, list):
                history = []
            prompt = (v.get("character_prompt") or "").strip() or DEFAULT_PERSONA
            if prompt == "You are a helpful assistant.":
                prompt = DEFAULT_PERSONA
            try:
                cid = int(k)
            except (TypeError, ValueError):
                continue
            last_snap = v.get("last_snap_entry_count")
            if last_snap is not None and not isinstance(last_snap, int):
                last_snap = None
            out[cid] = {"history": history, "character_prompt": prompt, "last_snap_entry_count": last_snap}
        return out
    except Exception:
        return {}


def _save_states() -> None:
    """Write user_states to JSON (chat_id as string keys)."""
    try:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        raw = {}
        for cid, state in user_states.items():
            raw[str(cid)] = {
                "history": state.get("history", []),
                "character_prompt": state.get("character_prompt") or "",
                "last_snap_entry_count": state.get("last_snap_entry_count"),
            }
        with open(TELEGRAM_STATES_FILE, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=0)
    except Exception:
        pass


from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

user_states = {}
user_states.update(_load_states())

TELEGRAM_MAX_MESSAGE_LENGTH = 4096
TELEGRAM_MAX_CAPTION_LENGTH = 1024


async def send_text_chunked(bot, chat_id: int, text: str) -> None:
    """Send text in chunks if it exceeds Telegram's 4096 char limit."""
    if not text:
        return
    for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
        await bot.send_message(chat_id, text[i : i + TELEGRAM_MAX_MESSAGE_LENGTH])


def get_state(chat_id: int):
    if chat_id not in user_states:
        user_states[chat_id] = {"history": [], "character_prompt": DEFAULT_PERSONA, "last_snap_entry_count": None}
    return user_states[chat_id]


def _history_to_entries(history: list) -> list[dict]:
    """Convert [(user, asst), ...] to [{user, assistant}, ...] for run_snap."""
    return [{"user": u, "assistant": a} for u, a in history]


def _run_snap_to_completion(entries: list, character_prompt: str, xai_key: str, last_snap_entry_count: int | None = None):
    """Sync: consume run_snap, return (gallery_list, story, error_msg, new_snap_count). error_msg is None on success; new_snap_count is len(entries) on success for caller to persist."""
    last_status = ""
    for gallery_list, status_msg, _, story in run_snap(
        [], character_prompt, "grok", xai_key, entries=entries, last_snap_entry_count=last_snap_entry_count
    ):
        last_status = status_msg or last_status
        if story and len(gallery_list) >= _dp.SNAP_NUM_IMAGES:
            return (gallery_list, story, None, len(entries))
    return ([], "", last_status or "Snapshot generation failed.", None)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Chat with me (memory & skills enabled), or use /snap to generate a Visual Snapshot from our conversation."
    )


async def cmd_snap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = get_state(chat_id)

    if not state["history"]:
        await update.message.reply_text("No conversation yet. Chat a few turns first, then try /snap.")
        return

    await update.message.reply_text("Generating snapshot… 生成中，可继续聊天。")

    bot = context.bot
    entries = _history_to_entries(state["history"])
    character_prompt = state["character_prompt"] or ""
    xai_key = os.environ.get("XAI_API_KEY", "").strip()

    async def _do_snap_and_send():
        try:
            gallery_list, story, error_msg, new_snap_count = await asyncio.to_thread(
                _run_snap_to_completion, entries, character_prompt, xai_key, state.get("last_snap_entry_count")
            )
            if error_msg:
                await send_text_chunked(bot, chat_id, error_msg)
            else:
                if new_snap_count is not None:
                    state["last_snap_entry_count"] = new_snap_count
                    _save_states()
                for path, caption in gallery_list:
                    if path and os.path.isfile(path):
                        cap = (caption or "")[:TELEGRAM_MAX_CAPTION_LENGTH]
                        with open(path, "rb") as f:
                            await bot.send_photo(chat_id, photo=f, caption=cap)
                await send_text_chunked(bot, chat_id, story)
        except Exception as e:
            await send_text_chunked(bot, chat_id, f"Snapshot failed: {e!s}")

    asyncio.create_task(_do_snap_and_send())


def _clear_global_memory() -> str:
    """Clear MEMORY.md, memory/*.md, and .memory_index.sqlite. Return '' on success, error message on failure."""
    try:
        workspace_dir = getattr(_cb, "WORKSPACE_DIR", None)
        if workspace_dir is None:
            return ""
        workspace_dir = Path(workspace_dir)
        mem_md = workspace_dir / "MEMORY.md"
        if mem_md.exists():
            mem_md.write_text("", encoding="utf-8")
        memory_dir = workspace_dir / "memory"
        if memory_dir.is_dir():
            for f in memory_dir.glob("*.md"):
                try:
                    f.unlink()
                except Exception:
                    pass
        index_path = Path(getattr(_cb, "MEMORY_INDEX_PATH", "") or (workspace_dir / ".memory_index.sqlite"))
        if index_path.exists():
            index_path.unlink()
        if hasattr(_cb, "_MEMORY_FLUSH_CURSOR_TURNS"):
            _cb._MEMORY_FLUSH_CURSOR_TURNS = 0
        return ""
    except Exception as e:
        return str(e)


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear this chat's history and global long-term memory."""
    chat_id = update.effective_chat.id
    state = get_state(chat_id)
    state["history"] = []
    state["character_prompt"] = DEFAULT_PERSONA
    _save_states()

    err = _clear_global_memory()
    if err:
        await send_text_chunked(context.bot, chat_id, f"对话历史已清空。长期记忆清除时出错：{err}")
    else:
        await update.message.reply_text("已清空本对话历史与长期记忆。")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_msg = (update.message.text or "").strip()
    if not user_msg:
        return

    state = get_state(chat_id)
    await context.bot.send_chat_action(chat_id, "typing")

    use_vector = os.environ.get("MEMORY_SEARCH_ENABLED", "").strip().lower() in ("1", "true", "yes")
    enable_skill_exec = os.environ.get("SKILL_EXEC_ENABLED", "").strip().lower() in ("1", "true", "yes")

    full_reply = ""
    history_for_state = list(state["history"])

    for h, _m, _s, wav_path in stream_chat(
        state["character_prompt"],
        state["history"],
        user_msg,
        enable_tts=False,
        tts_api_key="",
        tts_backend="dashscope",
        chat_backend="grok",
        xai_api_key=os.environ.get("XAI_API_KEY", ""),
        use_memory=True,
        use_vector_search=use_vector,
        enable_skill_exec=enable_skill_exec,
    ):
        if h:
            pairs = _history_to_tuples(h)
            if pairs:
                last_user, last_asst = pairs[-1]
                full_reply = last_asst or ""
                history_for_state = [(u, a) for u, a in pairs]

    state["history"] = history_for_state
    _append_conversation_log(state["character_prompt"], user_msg, full_reply)
    _save_states()

    await send_text_chunked(context.bot, chat_id, full_reply)


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")

    t = threading.Thread(target=_run_config_browser, daemon=True)
    t.start()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("snap", cmd_snap))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
