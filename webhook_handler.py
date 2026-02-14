"""
Handle Telegram webhook updates: sync workspace/state from Blob, run handlers, persist back.
"""
import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

from storage import storage_read, storage_write, storage_read_text, storage_write_text, storage_read_json, storage_write_json


def _states_rel(slot_id: str) -> str:
    return f"states/{slot_id}/qx_states.json"


def _workspace_rel(slot_id: str, *parts: str) -> str:
    return "workspace/" + slot_id + ("/" + "/".join(parts) if parts else "")


def _sync_workspace_from_blob(user_id: str, slot_id: str, tmp_base: Path) -> Path:
    """Copy workspace/{slot_id}/ from Blob to tmp_base/workspace/{slot_id}. Return tmp workspace path."""
    slot_dir = tmp_base / "workspace" / slot_id
    memory_dir = slot_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    mem_md_rel = _workspace_rel(slot_id, "MEMORY.md")
    data = storage_read(user_id, mem_md_rel)
    if data is not None:
        (slot_dir / "MEMORY.md").write_bytes(data)
    from datetime import datetime, timedelta
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    for day in (today, yesterday):
        rel = _workspace_rel(slot_id, "memory", f"{day}.md")
        data = storage_read(user_id, rel)
        if data is not None:
            (memory_dir / f"{day}.md").write_bytes(data)
    return slot_dir


def _sync_workspace_to_blob(user_id: str, slot_id: str, tmp_workspace: Path) -> None:
    """Write tmp workspace MEMORY.md and memory/*.md back to Blob."""
    mem_md = tmp_workspace / "MEMORY.md"
    if mem_md.exists():
        storage_write(user_id, _workspace_rel(slot_id, "MEMORY.md"), mem_md.read_bytes())
    memory_dir = tmp_workspace / "memory"
    if memory_dir.is_dir():
        for f in memory_dir.glob("*.md"):
            rel = _workspace_rel(slot_id, "memory", f.name)
            storage_write(user_id, rel, f.read_bytes())


def _load_state(user_id: str, slot_id: str, default_persona: str) -> dict:
    """Load qx_states.json from Blob. Return {chat_id: {history, character_prompt}} with int keys."""
    data = storage_read_json(user_id, _states_rel(slot_id), {})
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        history = v.get("history")
        if not isinstance(history, list):
            history = []
        prompt = (v.get("character_prompt") or "").strip() or default_persona
        try:
            cid = int(k)
        except (TypeError, ValueError):
            continue
        out[cid] = {"history": history, "character_prompt": prompt}
    return out


def _save_state(user_id: str, slot_id: str, state: dict) -> None:
    raw = {}
    for cid, s in state.items():
        raw[str(cid)] = {"history": s.get("history", []), "character_prompt": s.get("character_prompt") or ""}
    storage_write_json(user_id, _states_rel(slot_id), raw)


TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def _chat_id_from_update(update_data: dict) -> int | None:
    """Extract chat_id from Telegram Update dict."""
    msg = update_data.get("message") or update_data.get("edited_message")
    if msg:
        chat = msg.get("chat")
        if isinstance(chat, dict) and "id" in chat:
            return chat["id"]
    cb = update_data.get("callback_query")
    if isinstance(cb, dict) and isinstance(cb.get("message"), dict):
        chat = cb["message"].get("chat")
        if isinstance(chat, dict) and "id" in chat:
            return chat["id"]
    return None


def _send_error_fallback(token: str, update_data: dict, error_detail: str) -> None:
    """Send a short error message to the user so they get some response. Runs sync."""
    chat_id = _chat_id_from_update(update_data)
    if not chat_id or not (token or "").strip():
        return
    msg = (
        "Reply failed. Check your Vercel deployment logs. "
        "Common cause: set XAI_API_KEY (and optionally GROK_CHAT_MODEL) in Vercel Environment Variables."
    )
    if len(error_detail) < 100:
        msg += f" Error: {error_detail}"
    try:
        from telegram import Bot
        bot = Bot(token=token.strip())
        asyncio.run(bot.send_message(chat_id=chat_id, text=msg[:4000]))
    except Exception:
        pass


async def _send_text_chunked(bot, chat_id: int, text: str) -> None:
    if not text:
        return
    for i in range(0, len(text), TELEGRAM_MAX_MESSAGE_LENGTH):
        await bot.send_message(chat_id, text[i : i + TELEGRAM_MAX_MESSAGE_LENGTH])


def handle_webhook_update(user_id: str, slot_id: str, agent: dict, update_data: dict) -> None:
    """Process one Telegram Update. Sync workspace/state, run handlers, persist. Blocking."""
    token = (agent.get("telegram_bot_token") or "").strip()
    if not token:
        return
    persona_text = (agent.get("persona_text") or "").strip() or "You are a helpful assistant."
    request_id = str(uuid.uuid4())[:8]
    tmp_base = Path(tempfile.gettempdir()) / f"webhook_{request_id}"
    tmp_base.mkdir(parents=True, exist_ok=True)
    try:
        tmp_workspace = _sync_workspace_from_blob(user_id, slot_id, tmp_base)
        os.environ["CHATBOT_WORKSPACE"] = str(tmp_workspace)
        for key in ("XAI_API_KEY", "GROK_CHAT_MODEL", "OPENAI_API_KEY", "DASHSCOPE_API_KEY", "SKILL_EXEC_ENABLED", "MEMORY_SEARCH_ENABLED"):
            if key in os.environ:
                pass
            # Already in env from Vercel
        user_states = _load_state(user_id, slot_id, persona_text)

        from telegram import Update, Bot
        from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

        bot = Bot(token=token)

        def get_state(chat_id: int):
            if chat_id not in user_states:
                user_states[chat_id] = {"history": [], "character_prompt": persona_text}
            return user_states[chat_id]

        async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text(
                "Hi! Chat with me (memory enabled), or use /clear to reset. /snap is not available in webhook mode."
            )

        async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
            chat_id = update.effective_chat.id
            state = get_state(chat_id)
            state["history"] = []
            state["character_prompt"] = persona_text
            _clear_workspace_memory(tmp_workspace)
            await update.message.reply_text("已清空本对话历史与长期记忆。")

        async def cmd_snap(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Visual Snapshot is not available in Vercel webhook mode. Use local run for /snap.")

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
            chat_id = update.effective_chat.id
            user_msg = (update.message.text or "").strip()
            if not user_msg:
                return
            state = get_state(chat_id)
            await context.bot.send_chat_action(chat_id, "typing")
            use_vector = os.environ.get("MEMORY_SEARCH_ENABLED", "").strip().lower() in ("1", "true", "yes")
            enable_skill = os.environ.get("SKILL_EXEC_ENABLED", "").strip().lower() in ("1", "true", "yes")
            full_reply = ""
            try:
                from chat_core import stream_chat, _history_to_tuples, _append_conversation_log
                for h, _m, _s, _wav in stream_chat(
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
                    enable_skill_exec=enable_skill,
                ):
                    if h:
                        pairs = _history_to_tuples(h)
                        if pairs:
                            last_user, last_asst = pairs[-1]
                            full_reply = last_asst or ""
                            state["history"] = [(u, a) for u, a in pairs]
                _append_conversation_log(state["character_prompt"], user_msg, full_reply)
                await _send_text_chunked(context.bot, chat_id, full_reply or "（无回复）")
            except Exception as e:
                err_msg = str(e)[:500] if str(e) else "Unknown error"
                fallback = (
                    "Reply failed. If on Vercel, set XAI_API_KEY in Project Settings → Environment Variables. "
                    f"Error: {err_msg}"
                )
                await _send_text_chunked(context.bot, chat_id, fallback[:4000])

        def _clear_workspace_memory(workspace_dir: Path) -> None:
            try:
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
            except Exception:
                pass

        app = Application.builder().token(token).build()
        app.add_handler(CommandHandler("start", cmd_start))
        app.add_handler(CommandHandler("snap", cmd_snap))
        app.add_handler(CommandHandler("clear", cmd_clear))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        update_obj = Update.de_json(update_data, bot)

        async def run_update():
            await app.initialize()
            try:
                await app.process_update(update_obj)
            finally:
                await app.shutdown()

        asyncio.run(run_update())

        _save_state(user_id, slot_id, user_states)
        _sync_workspace_to_blob(user_id, slot_id, tmp_workspace)
    finally:
        try:
            import shutil
            if tmp_base.exists():
                shutil.rmtree(tmp_base, ignore_errors=True)
        except Exception:
            pass
