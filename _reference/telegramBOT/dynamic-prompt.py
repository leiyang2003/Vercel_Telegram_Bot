"""
Character Chat Web UI — chat with a character via mlx_lm.server (OpenAI-compatible API).
Start the server separately: mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-4bit --port 8000
TTS: DashScope CosyVoice (streaming); voice longanhuan. Set DASHSCOPE_API_KEY in .env or enter in UI.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env: 先尝试当前工作目录，再加载脚本所在目录（脚本目录优先）
_env_dir = Path(__file__).resolve().parent
load_dotenv()  # cwd 下的 .env
load_dotenv(_env_dir / ".env")  # Local-Model/.env
# #region agent log
def _dlog(msg, data=None, hypothesisId=None):
    try:
        import time as _t
        o = {"timestamp": int(_t.time() * 1000), "location": "local-ml.py", "message": msg, "runId": "run1"}
        if data is not None: o["data"] = data
        if hypothesisId is not None: o["hypothesisId"] = hypothesisId
        with open("/Users/leiyang/Desktop/Coding/.cursor/debug.log", "a", encoding="utf-8") as f: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    except Exception: pass
_env_file = _env_dir / ".env"
_dlog("after load_dotenv", {"script_dir": str(_env_dir), "env_file_exists": _env_file.exists(), "cwd": str(Path.cwd()), "cwd_env_exists": (Path.cwd() / ".env").exists(), "DASHSCOPE_API_KEY_set": bool((os.environ.get("DASHSCOPE_API_KEY") or "").strip())}, "H1")
# #endregion

import base64
import html as html_module
import queue
import shutil
import re
import struct
import subprocess
import tempfile
import threading
from datetime import datetime
import gradio as gr
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_SYSTEM = "You are a helpful assistant."
MAX_TOKENS = 512
TEMPERATURE = 0.7

# --- Chat: Grok AI (xAI) ---
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
GROK_CHAT_MODEL = os.environ.get("GROK_CHAT_MODEL", "grok-3")


def _get_chat_config(backend: str, xai_key: str = "") -> tuple[str, dict]:
    """Return (url, headers) for chat completion. backend: 'lan' | 'grok'."""
    if (backend or "").strip().lower() == "grok":
        key = (xai_key or "").strip() or (os.environ.get("XAI_API_KEY") or "").strip()
        return XAI_CHAT_URL, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    return API_URL, {"Content-Type": "application/json"}


def _extract_api_error_message(exc: requests.RequestException, default: str) -> str:
    """Extract error message from API response; supports OpenAI-style and plain message/detail."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None) if resp is not None else None
    if resp is None:
        # Connection error, timeout, DNS, etc.
        _dlog("Grok request no response", {"error": str(type(exc).__name__), "message": str(exc)}, "grok_err")
        return "Grok 连接失败（未收到响应），请检查网络、代理或 api.x.ai 是否可访问。"
    # Force-read body when streaming so we get the error payload
    try:
        _ = getattr(resp, "content", None) or getattr(resp, "text", None)
    except Exception:
        pass
    text = (getattr(resp, "text", None) or "").strip()
    if not text and getattr(resp, "content", None):
        try:
            text = (resp.content or b"").decode("utf-8", errors="replace").strip()
        except Exception:
            pass
    _dlog("Grok API error response", {"status": status, "text_preview": (text or "")[:600]}, "grok_err")
    try:
        err = resp.json()
        if isinstance(err, dict):
            msg = (err.get("error") or {})
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
    status_str = str(status) if status is not None else "?"
    if text:
        return f"[{status_str}] {text[:200]}..." if len(text) > 200 else f"[{status_str}] {text}"
    return f"[{status_str}] 无错误详情，请查看 .cursor/debug.log 中 hypothesisId=grok_err 的条目。"


# --- UI: display name for chat header (set DISPLAY_MODEL_NAME in env to override) ---
DISPLAY_MODEL_NAME = os.environ.get("DISPLAY_MODEL_NAME", "MLX / Qwen")
DISPLAY_SUBTITLE = os.environ.get("DISPLAY_SUBTITLE", "LOCAL MLX SERVER")
APP_HEADER_TITLE = os.environ.get("APP_HEADER_TITLE", "Character Chat")

# --- UI: custom CSS for reference design (purple accents, chat bubbles) ---
# 宋体细体 font stack (Songti SC Light / 细字)
_FONT_SONGTI_LIGHT = '"Songti SC Light", "STSongti-SC-Light", "SimSun", "NSimSun", "宋体", "Songti SC", "STSong", serif'
UI_CSS = f"""
/* Purple primary accents */
.primary-btn, .gr-button-primary, button.primary {{ background: #7c3aed !important; color: white !important; border: none !important; }}
.primary-btn:hover, .gr-button-primary:hover {{ background: #6d28d9 !important; }}
/* Chat: user messages right, purple; assistant left, grey */
.gr-chatbot .message.user {{ margin-left: auto; max-width: 85%; background: #7c3aed !important; color: white !important; border-radius: 1rem 1rem 0.25rem 1rem !important; }}
.gr-chatbot .message.bot, .gr-chatbot .message:not(.user) {{ margin-right: auto; max-width: 85%; background: #f3f4f6 !important; color: #1f2937 !important; border-radius: 1rem 1rem 1rem 0.25rem !important; }}
/* Load persona file link-style */
.gr-file .wrap {{ border: none !important; }}
.gr-file label, .gr-file .label {{ color: #7c3aed !important; text-decoration: underline; cursor: pointer; }}
/* Section spacing */
.gr-form, .gr-box {{ border-radius: 0.5rem; }}
.gr-block {{ margin-bottom: 0.75rem; }}
/* 满屏设计：占满视口宽高 */
.gradio-container {{ font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; width: 100% !important; max-width: none !important; min-height: 100vh !important; padding: 0 !important; box-sizing: border-box; }}
.gradio-container .main {{
  min-height: calc(100vh - 0px) !important;
  max-width: none !important;
}}
.gradio-container .contain {{ max-width: none !important; }}
/* Main page: 宋体细字 for all Chinese text */
/* Header/footer text */
.ui-header {{ font-size: 1.125rem; font-weight: 600; color: #374151; margin-bottom: 0.25rem; font-family: {_FONT_SONGTI_LIGHT}; }}
.ui-subtitle {{ font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
.ui-footer {{ font-size: 0.75rem; color: #9ca3af; text-align: center; margin-top: 0.5rem; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
/* Visual Snapshot card */
#visual-snapshot-card {{ border-radius: 0.75rem; padding: 1rem; border: 1px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.08); background: #fafafa; }}
.snap-card-title {{ font-size: 1.25rem; font-weight: 600; color: #1f2937; margin-bottom: 0.25rem; font-family: {_FONT_SONGTI_LIGHT}; }}
.snap-card-subtitle {{ font-size: 0.8rem; color: #6b7280; margin-bottom: 0.75rem; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
.snap-story-container {{ max-height: 420px; overflow-y: auto; display: block; width: 100%; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
.snap-dropcap {{ float: left; font-size: 3em; line-height: 1; color: #7c3aed; margin-right: 0.25rem; font-weight: 600; font-family: {_FONT_SONGTI_LIGHT}; }}
.snap-story-body {{ font-size: 0.95rem; line-height: 1.6; color: #374151; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
.snap-article p {{ overflow: auto; display: block; width: 100%; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
.snap-story-placeholder {{ color: #9ca3af; font-style: italic; font-family: {_FONT_SONGTI_LIGHT}; }}
.snap-illus {{ margin: 1.25em 0; text-align: center; clear: both; display: block; width: 100%; }}
.snap-illus img {{ max-width: 100%; height: auto; border-radius: 0.5rem; display: block; margin: 0 auto; }}
.snap-illus .caption {{ font-size: 0.85rem; color: #6b7280; margin-top: 0.35rem; font-family: {_FONT_SONGTI_LIGHT}; font-weight: 300; }}
"""

# --- TTS (DashScope CosyVoice) ---
TTS_MAX_SEGMENT_CHARS = 2000   # CosyVoice per-call limit
TTS_SENTENCE_MAX_CHARS = 200   # max chars per segment when no sentence end found
TTS_SAMPLE_RATE = 22050        # PCM playback rate for CosyVoice default
_SCRIPT_DIR = Path(__file__).resolve().parent
TTS_OUTPUT_WAV = str(_SCRIPT_DIR / "last_tts.wav")  # 固定路径，界面可点击播放

# 局域网 TTS（MLX 兼容接口）
LAN_TTS_DEFAULT_URL = "http://192.168.31.134:8000/v1/audio/speech"
LAN_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
LAN_TTS_MAX_SEGMENT_CHARS = 800   # 单段上限，减少切碎
LAN_TTS_SENTENCE_MAX_CHARS = 400  # 无句末时截断长度

# --- Conversation log (for 调整方向 analysis) ---
CHAT_LOG_DIR = _SCRIPT_DIR / "chat_logs"
CONVERSATIONS_LOG = CHAT_LOG_DIR / "conversations.jsonl"
CONVERSATION_PROMPT_TRUNCATE = 2000  # max chars of character_prompt to store per turn
ANALYSIS_SYSTEM_PROMPT = (
    "你是一个人设优化助手。根据以下对话记录，列出 3–5 条具体、可操作的「调整方向」，"
    "用于修改角色人设/系统提示词以更符合用户期望。每条一行，简短明确。只输出调整建议，不要其他解释。"
)
ANALYSIS_MAX_TURNS = 30   # use last N turns for analysis
ANALYSIS_MAX_CHARS = 12000  # max total chars of dialogue to send to model

# --- Snap: last N turns -> image prompt -> xAI Imagine ---
SNAP_MAX_TURNS = 10
SNAP_LOG_DIR = CHAT_LOG_DIR / "snaps"
VSNAPSHOT_DIR = _SCRIPT_DIR / "VSnapshot"
SNAP_SUMMARY_PROMPT = (
    "结合上述角色人设与对话，总结为一个适合图像生成的英文提示词。"
    "描述场景、人物、氛围和风格，200词以内，仅输出提示词，不要其他解释。"
)
SNAP_SCENE_SYSTEM = (
    "根据对话内容总结当前场景，生成适合图像生成的提示词。"
    "尽量描述细节：包括场景、环境、人物外貌、情绪、表情、姿势与动作、光线、构图、质感、氛围和风格等，使画面具体可感。仅输出提示词，不要其他解释。"
)
SNAP_SAFETY_REWRITE_PROMPT = (
    "You are a content-safety rewriter for image generation prompts. "
    "Rewrite the given prompt to remove any sensitive, violent, adult, or policy-violating content. "
    "Keep the scene, mood, and style suitable for a general audience. "
    "Output only the revised English prompt, no explanation or extra text."
)
# Story: two modes — last_n (last N turns) or since_snap (since last /snap, previous story as context)
SNAP_STORY_MODE = os.environ.get("SNAP_STORY_MODE", "since_snap").strip().lower()
SNAP_STORY_FOCUS_TURNS = 10  # last_n 模式下使用
SNAP_MAX_TURNS_SINCE_SNAP = 30  # since_snap 时从文件读取的最大轮数，以便正确切片
SNAP_STATE_FILE = CHAT_LOG_DIR / "snap_state.json"
SNAP_STORY_SYSTEM = (
    "你是一个故事写作者。对话记录分为两部分：「此前的全部对话」仅作背景参考；「最近10轮对话」是本次故事的主要依据。"
    "请主要根据「最近10轮对话」写出一段完整、连贯的叙事故事，可结合此前对话的背景信息，但避免简单复述或与前文重复。"
    "要求：500-1000 字，有开头、发展、结尾，语言流畅。只输出故事正文，不要标题、序号或其它解释。"
)
SNAP_STORY_SYSTEM_SINCE_SNAP = (
    "你是一个故事写作者。以下提供「上一次已写的故事」作为背景；「自上次 snapshot 至今的对话」是本次续写的主要依据。"
    "请根据这段新对话写出一段完整、连贯的叙事续篇，与上一次故事衔接自然，可沿用背景与人物，但避免简单复述。"
    "要求：500-1000 字，有开头、发展、结尾，语言流畅。只输出故事正文，不要标题、序号或其它解释。"
)
SNAP_CHARACTER_DESIGN_FILE = CHAT_LOG_DIR / "snap_character_design.txt"
# Number of images to generate (configurable via SNAP_NUM_IMAGES env, default 3)
SNAP_NUM_IMAGES = max(1, min(6, int(os.environ.get("SNAP_NUM_IMAGES", "3"))))
# When no saved design: output (1 + SNAP_NUM_IMAGES) lines (line 0 = character design, lines 1-N = scene prompts)
SNAP_THREE_ILLUST_SYSTEM_FOUR_LINES = (
    f"You are an expert at writing image-generation prompts. Given a story, output exactly {1 + SNAP_NUM_IMAGES} lines. "
    "Line 0 (character design): Describe each main character's figure (身材) and appearance/face (容貌) in one block of English; "
    "for each character clearly state nationality (which country, e.g. Chinese, Japanese, American) and age (年纪). "
    "Also include body type, height, face, hair, and clothing. "
    f"Lines 1-{SNAP_NUM_IMAGES}: {SNAP_NUM_IMAGES} illustration prompts. Each prompt must describe a single moment with at most ONE or TWO characters visibly in frame; do not put all characters in every image. "
    f"Each of lines 1-{SNAP_NUM_IMAGES} must be one line, detailed (scene, pose, mood, lighting). State clearly who is in frame (e.g. only the woman; or the woman and the man). "
    f"Use the same visual style for all {SNAP_NUM_IMAGES}. Output format: exactly {1 + SNAP_NUM_IMAGES} lines, no numbering, no extra text."
)
# When saved design exists: output SNAP_NUM_IMAGES lines only (scene prompts that use the given character design)
SNAP_THREE_ILLUST_SYSTEM_THREE_LINES = (
    "You are an expert at writing image-generation prompts. Use the character design provided below for every character "
    f"(same figure and appearance in all prompts). Output exactly {SNAP_NUM_IMAGES} lines: the {SNAP_NUM_IMAGES} illustration prompts. "
    "Each prompt must describe a single moment with at most ONE or TWO characters visibly in frame; do not put all characters in every image. "
    "State clearly who is in frame (e.g. only Autumn Frost; or Autumn Frost and the Master). Only scene, pose, and expression may change. "
    f"Each line: one illustration prompt (scene/moment, mood, lighting). Same visual style for all {SNAP_NUM_IMAGES}. No numbering."
)
SNAP_IMAGE_STYLE = (
    "Digital illustration, soft lighting, consistent color palette and painterly texture, "
    "same character design and appearance across all scenes, consistent face and clothing."
)
XAI_IMAGINE_URL = "https://api.x.ai/v1/images/generations"
XAI_IMAGINE_MODEL = "grok-imagine-image"


def _append_conversation_log(character_prompt: str, user_msg: str, assistant_reply: str) -> None:
    """Append one turn to conversations.jsonl. Creates chat_logs dir if needed."""
    if not (user_msg or assistant_reply):
        return
    try:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        prompt_snippet = (character_prompt or "").strip()
        if len(prompt_snippet) > CONVERSATION_PROMPT_TRUNCATE:
            prompt_snippet = prompt_snippet[:CONVERSATION_PROMPT_TRUNCATE] + "..."
        import time as _t
        record = {
            "timestamp": _t.time(),
            "character_prompt": prompt_snippet,
            "user": user_msg.strip(),
            "assistant": (assistant_reply or "").strip(),
        }
        with open(CONVERSATIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _rule_based_prompt_ideas(entries: list) -> str:
    """Fallback: heuristics over user messages to suggest prompt adjustments."""
    lines = []
    correction_phrases = ("不对", "不是", "应该是", "别这样", "错了", "不是这样", "别这么说")
    length_short = ("太长了", "简短点", "短一点", "简短些", "不要太长", "精简")
    length_long = ("多说点", "详细点", "展开说说", "再详细")
    tone_formal = ("别这么正式", "随意点", "轻松点", "别端着")
    for e in entries:
        user = (e.get("user") or "").strip()
        if not user:
            continue
        u_lower = user.lower()
        for p in correction_phrases:
            if p in user or p in u_lower:
                lines.append("· 用户曾纠正回复内容，可在人设中更明确相关设定或禁忌。")
                break
        for p in length_short:
            if p in user:
                lines.append("· 用户希望回复更短，可在人设中增加「简短回应」「少说废话」等指示。")
                break
        for p in length_long:
            if p in user:
                lines.append("· 用户希望回复更详细，可在人设中允许或鼓励展开说明。")
                break
        for p in tone_formal:
            if p in user:
                lines.append("· 用户希望语气更随意，可在人设中强调口语化、轻松风格。")
                break
    if not lines:
        return "（根据当前对话未检测到明显的纠正或偏好，可多聊几轮后再分析，或使用 MLX 服务器做更细分析。）"
    return "\n".join(dict.fromkeys(lines))  # dedupe while preserving order


def analyze_log_for_prompt_ideas(log_path: Path, chat_backend: str = "lan", xai_api_key: str = "") -> str:
    """Read conversation log, call chat API (LAN or Grok) for 3–5 调整方向; on failure use rule-based fallback."""
    if not log_path or not log_path.exists() or log_path.stat().st_size == 0:
        return "暂无对话记录，请先进行几轮对话。"
    entries = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return "无法读取对话记录。"
    if not entries:
        return "暂无对话记录，请先进行几轮对话。"
    # Use last N turns and cap total chars
    recent = entries[-ANALYSIS_MAX_TURNS:] if len(entries) > ANALYSIS_MAX_TURNS else entries
    parts = []
    total = 0
    for e in reversed(recent):
        user = (e.get("user") or "").strip()
        asst = (e.get("assistant") or "").strip()
        block = f"用户：{user}\n助手：{asst}\n"
        if total + len(block) > ANALYSIS_MAX_CHARS:
            break
        parts.append(block)
        total += len(block)
    parts.reverse()
    dialogue = "\n".join(parts).strip()
    if not dialogue:
        return "对话内容为空，请先进行几轮对话。"
    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    try:
        payload = {
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": "对话记录：\n\n" + dialogue},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": False,
        }
        if (chat_backend or "").strip().lower() == "grok":
            payload["model"] = GROK_CHAT_MODEL
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if content and content.strip():
            return content.strip()
    except Exception:
        pass
    return _rule_based_prompt_ideas(entries)


def _get_last_n_turns(n: int, entries: list[dict] | None = None, log_path: Path | None = None) -> list[dict]:
    """Read conversation log and return last n entries. If entries is provided, use it directly. If log_path is provided, read from that file. Otherwise use CONVERSATIONS_LOG."""
    if entries is not None:
        return entries[-n:] if len(entries) > n else entries
    path = log_path if log_path is not None else CONVERSATIONS_LOG
    if not path.exists() or path.stat().st_size == 0:
        return []
    entries_out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries_out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return entries_out[-n:] if len(entries_out) > n else entries_out


def _summarize_to_image_prompt(entries: list[dict], character_prompt: str = "", chat_backend: str = "lan", xai_api_key: str = "") -> str:
    """Ask the chat API to produce an image prompt for the current scene (scene-based, not character-based); return the reply as the image prompt."""
    if not entries:
        return ""
    parts = []
    for e in entries:
        user = (e.get("user") or "").strip()
        asst = (e.get("assistant") or "").strip()
        parts.append(f"用户：{user}\n助手：{asst}")
    dialogue = "\n\n".join(parts).strip()
    if not dialogue:
        return ""
    user_content = "对话记录：\n\n" + dialogue + "\n\n把现在的场景做一个生图Prompt"
    messages = [{"role": "system", "content": SNAP_SCENE_SYSTEM}, {"role": "user", "content": user_content}]
    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    try:
        payload = {
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": False,
        }
        if (chat_backend or "").strip().lower() == "grok":
            payload["model"] = GROK_CHAT_MODEL
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return (content or "").strip() or ""
    except Exception:
        return ""


def _dialog_and_prompt_to_story(
    entries: list[dict],
    character_prompt: str = "",
    chat_backend: str = "lan",
    xai_api_key: str = "",
    *,
    last_snap_entry_count: int | None = None,
    previous_story: str = "",
) -> str:
    """Build dialogue from entries; call chat API for one story (500-1000 字). Supports last_n or since_snap mode. Return "" on failure."""
    if not entries:
        return ""

    def format_dialogue(ents: list[dict]) -> str:
        parts = []
        for e in ents:
            user = (e.get("user") or "").strip()
            asst = (e.get("assistant") or "").strip()
            parts.append(f"用户：{user}\n助手：{asst}")
        return "\n\n".join(parts).strip()

    use_since_snap = (
        SNAP_STORY_MODE == "since_snap"
        and last_snap_entry_count is not None
        and (previous_story or "").strip()
        and 0 <= last_snap_entry_count < len(entries)
    )
    if use_since_snap:
        focus_entries = entries[last_snap_entry_count:]
        focus_dialogue = format_dialogue(focus_entries)
        if not focus_dialogue:
            use_since_snap = False
    if not use_since_snap:
        n_focus = min(SNAP_STORY_FOCUS_TURNS, len(entries))
        focus_entries = entries[-n_focus:]
        context_entries = entries[:-n_focus] if len(entries) > n_focus else []
        focus_dialogue = format_dialogue(focus_entries)
        context_dialogue = format_dialogue(context_entries) if context_entries else ""

    if not focus_dialogue:
        return ""
    persona = (character_prompt or "").strip()
    user_content = "角色/背景人设：\n\n" + (persona or "（无）") + "\n\n"
    if use_since_snap:
        user_content += "上一次已写的故事（仅作背景参考）：\n\n" + (previous_story or "").strip() + "\n\n"
        user_content += "自上次 snapshot 至今的对话（请主要依据此写续篇）：\n\n" + focus_dialogue + "\n\n请根据以上人设、上文故事与这段新对话，写出一段完整续篇（500-1000 字）。"
        system = SNAP_STORY_SYSTEM_SINCE_SNAP
    else:
        if context_dialogue:
            user_content += "此前的全部对话（仅作背景参考）：\n\n" + context_dialogue + "\n\n"
        user_content += f"最近{SNAP_STORY_FOCUS_TURNS}轮对话（请主要依据此写故事，避免与前文重复）：\n\n" + focus_dialogue + "\n\n请根据以上人设和对话，写出一段完整故事（500-1000 字）。"
        system = SNAP_STORY_SYSTEM
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    try:
        payload = {
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.5,
            "stream": False,
        }
        if (chat_backend or "").strip().lower() == "grok":
            payload["model"] = GROK_CHAT_MODEL
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return (content or "").strip() or ""
    except Exception:
        return ""


def _load_snap_state(state_path: Path | None = None) -> int | None:
    """Read last_snap_entry_count from snap_state.json. Returns None if missing or on error."""
    path = state_path if state_path is not None else SNAP_STATE_FILE
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            n = data.get("last_snap_entry_count")
            if n is not None and isinstance(n, int) and n >= 0:
                return n
    except Exception:
        pass
    return None


def _save_snap_state(last_snap_entry_count: int, state_path: Path | None = None) -> None:
    """Write last_snap_entry_count to snap_state.json."""
    path = state_path if state_path is not None else SNAP_STATE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"last_snap_entry_count": last_snap_entry_count}, f, ensure_ascii=False)
    except Exception:
        pass


def _load_snap_character_design() -> str:
    """Read persisted character design (身材+容貌) from file. Returns empty string if missing or on error."""
    if not SNAP_CHARACTER_DESIGN_FILE.exists():
        return ""
    try:
        with open(SNAP_CHARACTER_DESIGN_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _save_snap_character_design(text: str) -> None:
    """Persist character design to file for use in future Snap rounds."""
    if not (text or "").strip():
        return
    try:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SNAP_CHARACTER_DESIGN_FILE, "w", encoding="utf-8") as f:
            f.write(text.strip())
    except Exception:
        pass


def _strip_numbered_prefix(line: str) -> str:
    """Remove leading '1.', 'Prompt 1:', etc. from a line."""
    for sep in (". ", ": ", "、", " "):
        if sep in line and line.split(sep, 1)[0].replace(".", "").replace(":", "").strip().isdigit():
            return line.split(sep, 1)[1].strip()
    return line


def _story_to_three_image_prompts(
    story: str, chat_backend: str = "lan", xai_api_key: str = "", saved_character_design: str = ""
) -> tuple[list[str], str]:
    """
    Ask chat API for N illustration prompts and the character design to use.
    Returns (prompts list of SNAP_NUM_IMAGES strings, design_to_use string).
    When saved_character_design is non-empty: pass it to LLM, ask for N prompts only.
    When empty: ask for (1+N) lines (line 0 = design, lines 1-N = prompts), parse and return (prompts, new_design).
    """
    if not (story or "").strip():
        return ([], "")
    has_saved = bool((saved_character_design or "").strip())
    if has_saved:
        system = SNAP_THREE_ILLUST_SYSTEM_THREE_LINES
        user = f"Character design (use exactly this for all characters, same figure and appearance):\n\n{saved_character_design.strip()}\n\nStory:\n\n{(story or '').strip()}"
    else:
        system = SNAP_THREE_ILLUST_SYSTEM_FOUR_LINES
        user = "Story:\n\n" + (story or "").strip()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    try:
        payload = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
            "stream": False,
        }
        if (chat_backend or "").strip().lower() == "grok":
            payload["model"] = GROK_CHAT_MODEL
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        text = (content or "").strip()
        if not text:
            return ([], saved_character_design if has_saved else "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned = [_strip_numbered_prefix(ln) for ln in lines]
        n_img = SNAP_NUM_IMAGES
        if has_saved:
            # Expect n_img lines: the scene prompts
            if len(cleaned) >= n_img:
                return (cleaned[:n_img], saved_character_design)
            if len(cleaned) >= 1:
                # Pad with last prompt if LLM returned fewer
                pad = [cleaned[-1]] * (n_img - len(cleaned))
                return (cleaned[:n_img] + pad[:n_img - len(cleaned)], saved_character_design)
            return ([], saved_character_design)
        # No saved design: expect (1 + n_img) lines (design, prompt1, ... promptN)
        if len(cleaned) >= 1 + n_img:
            design_to_use = cleaned[0]
            return (cleaned[1 : 1 + n_img], design_to_use)
        if len(cleaned) >= 2:
            design_to_use = cleaned[0]
            prompts = cleaned[1:]
            pad = [prompts[-1]] * (n_img - len(prompts)) if len(prompts) < n_img else []
            return (prompts[:n_img] + pad, design_to_use)
        if len(cleaned) == 1:
            fallback = (story or "").strip()[:800]
            single = fallback if fallback else cleaned[0]
            return ([single] * n_img, single)
        fallback = (story or "").strip()[:800]
        return ([fallback] * n_img, fallback) if fallback else ([], "")
    except Exception:
        return ([], saved_character_design if has_saved else "")


def _rewrite_image_prompt_for_safety(prompt: str, chat_backend: str, xai_api_key: str) -> str:
    """Use chat API (LAN or Grok) to rewrite the image prompt, removing sensitive content. Returns rewritten prompt or \"\" on failure."""
    if not (prompt or "").strip():
        return ""
    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    try:
        payload = {
            "messages": [
                {"role": "system", "content": SNAP_SAFETY_REWRITE_PROMPT},
                {"role": "user", "content": (prompt or "").strip()},
            ],
            "max_tokens": 256,
            "temperature": 0.3,
            "stream": False,
        }
        if (chat_backend or "").strip().lower() == "grok":
            payload["model"] = GROK_CHAT_MODEL
        r = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return (content or "").strip() or ""
    except Exception:
        return ""


def _call_xai_imagine(prompt: str, api_key: str, filename_suffix: str = "") -> str | None:
    """
    Call xAI Imagine API; download image and save to SNAP_LOG_DIR.
    Returns local file path on success, None on failure.
    filename_suffix: optional suffix (e.g. "_1") to avoid overwriting when generating multiple images.
    """
    if not (prompt or prompt.strip()) or not (api_key or "").strip():
        return None
    key = api_key.strip()
    try:
        SNAP_LOG_DIR.mkdir(parents=True, exist_ok=True)
        body = {"model": XAI_IMAGINE_MODEL, "prompt": prompt.strip()}
        r = requests.post(
            XAI_IMAGINE_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json=body,
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        items = data.get("data") or []
        if not items:
            return None
        first = items[0]
        url = first.get("url")
        b64 = first.get("b64_json")
        if url:
            img_r = requests.get(url, timeout=60)
            img_r.raise_for_status()
            raw = img_r.content
        elif b64:
            raw = base64.b64decode(b64)
        else:
            return None
        if len(raw) < 100:
            return None
        import time as _t
        suffix = (filename_suffix or "").strip()
        fname = f"snap_{_t.strftime('%Y%m%d_%H%M%S', _t.localtime())}{suffix}.jpg"
        path = SNAP_LOG_DIR / fname
        with open(path, "wb") as f:
            f.write(raw)
        return str(path)
    except requests.RequestException as e:
        _dlog("xAI Imagine request failed", {"error": str(e)}, "xai_imagine")
        raise
    except Exception as e:
        _dlog("xAI Imagine error", {"error": str(type(e).__name__), "message": str(e)}, "xai_imagine")
        raise


def _format_imagine_error(e: Exception) -> str:
    """Build user-visible error message for xAI Imagine failure."""
    if isinstance(e, requests.RequestException):
        resp = getattr(e, "response", None)
        detail = ""
        if resp is not None:
            try:
                err = resp.json()
                detail = err.get("error", {}).get("message", resp.text[:200]) or resp.text[:200]
            except Exception:
                detail = resp.text[:200] if resp.text else str(resp.status_code)
        if not detail:
            detail = str(e)
        return f"xAI 图片生成失败：{detail}"
    return f"xAI 图片生成失败：{e!s}"


def _story_to_paragraphs(story: str) -> list:
    """Split story into paragraphs: by \\n\\n, then \\n; if no newlines, by sentence end (。！？) into ~3 chunks."""
    s = (story or "").strip()
    if not s:
        return []
    paras = [p.strip() for p in s.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in s.split("\n") if p.strip()]
    if len(paras) <= 1:
        parts = re.split(r"(?<=[。！？])", s)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) <= 1:
            return [s]
        n = min(3, len(parts))
        step = max(1, len(parts) // n)
        return ["".join(parts[i * step:(i + 1) * step]) for i in range(n - 1)] + ["".join(parts[(n - 1) * step:])]
    return paras


def _story_to_snapshot_html(story: str) -> str:
    """Build HTML for Visual Snapshot card: first character as purple drop cap, rest escaped."""
    if not (story or "").strip():
        return '<p class="snap-story-placeholder">No snapshot yet. Click Snap to generate.</p>'
    s = story.strip()
    first = s[0] if s else ""
    rest = html_module.escape(s[1:]) if len(s) > 1 else ""
    return f'<div class="snap-story-body"><p><span class="snap-dropcap">{first}</span>{rest}</p></div>'


def _build_article_html_with_illustrations(story: str, gallery_items: list, embed_images_as_base64: bool = False) -> str:
    """Build single-column article HTML: story segments with illustrations in between. If embed_images_as_base64, read image files and embed as data URLs for in-app display."""
    if not (story or "").strip() and not (gallery_items or []):
        return '<p class="snap-story-placeholder">No snapshot yet. Click Snap to generate.</p>'
    image_srcs = []
    if embed_images_as_base64:
        for item in gallery_items or []:
            path = item[0] if item and isinstance(item, (list, tuple)) and len(item) >= 1 and isinstance(item[0], str) else None
            if path and os.path.isfile(path):
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    b64 = base64.b64encode(raw).decode("ascii")
                    ext = (Path(path).suffix or ".jpg").lower()
                    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/jpeg"
                    image_srcs.append(f"data:{mime};base64,{b64}")
                except Exception:
                    image_srcs.append(None)
            else:
                image_srcs.append(None)
    else:
        image_srcs = []
        for item in gallery_items or []:
            if item and isinstance(item, (list, tuple)) and len(item) >= 1:
                image_srcs.append(html_module.escape(str(item[0])))
            elif isinstance(item, str):
                image_srcs.append(html_module.escape(item))
    n = len(image_srcs)
    if not (story or "").strip():
        parts = []
        for i, src in enumerate(image_srcs):
            if src:
                parts.append(f'<div class="snap-illus"><img src="{src}" alt="插图{i + 1}"><p class="caption">插图 {i + 1}</p></div>')
        return "\n".join(parts) if parts else '<p class="snap-story-placeholder">No snapshot yet.</p>'
    s = (story or "").strip()
    if n == 0:
        first, rest = s[0] if s else "", html_module.escape(s[1:]) if len(s) > 1 else ""
        return f'<div class="snap-story-body"><p><span class="snap-dropcap">{first}</span>{rest}</p></div>'
    paragraphs = _story_to_paragraphs(s)
    M = len(paragraphs)
    if M == 0:
        paragraphs = [s]
        M = 1
    after_para = [((k + 1) * (M + 1)) // (n + 1) - 1 for k in range(n)]
    parts = []
    for i, para in enumerate(paragraphs):
        if not para.strip():
            continue
        if i == 0:
            first, rest = para[0] if para else "", html_module.escape(para[1:]) if len(para) > 1 else ""
            parts.append(f'<p><span class="snap-dropcap">{first}</span>{rest}</p>')
        else:
            parts.append(f"<p>{html_module.escape(para)}</p>")
        for k in range(n):
            if after_para[k] == i and image_srcs[k]:
                parts.append(f'<div class="snap-illus"><img src="{image_srcs[k]}" alt="插图{k + 1}"><p class="caption">插图 {k + 1}</p></div>')
    return "\n".join(parts)


def _build_snapshot_page_html(story: str, image_filenames: list) -> str:
    """Build full standalone HTML: back link, story as article with illustrations embedded between segments."""
    back_link = '<a href="../index.html" style="display:inline-block;margin-bottom:1rem;color:#7c3aed;text-decoration:none;font-size:1rem;">← 返回总览</a>'
    css = """
    <style>
    body { font-family: "SimSun", "NSimSun", "宋体", "Songti SC", "STSong", serif; font-weight: 300; max-width: 900px; margin: 0 auto; padding: 1.5rem; color: #374151; }
    .snap-article { font-size: 1rem; line-height: 1.75; display: block; width: 100%; }
    .snap-dropcap { float: left; font-size: 3em; line-height: 1; color: #7c3aed; margin-right: 0.25rem; font-weight: 600; }
    .snap-article p { margin: 0 0 1em; overflow: auto; display: block; width: 100%; }
    .snap-illus { margin: 1.25em 0; text-align: center; clear: both; display: block; width: 100%; }
    .snap-illus img { max-width: 100%; height: auto; border-radius: 0.5rem; display: block; margin: 0 auto; }
    .snap-illus .caption { font-size: 0.85rem; color: #6b7280; margin-top: 0.35rem; font-family: "SimSun", "NSimSun", "宋体", "Songti SC", "STSong", serif; font-weight: 300; }
    .snap-story-placeholder { color: #9ca3af; font-style: italic; }
    </style>
    """
    if not (story or "").strip():
        body_parts = ['<p class="snap-story-placeholder">No snapshot yet.</p>']
        for i, fn in enumerate(image_filenames):
            body_parts.append(f'<div class="snap-illus"><img src="{html_module.escape(fn)}" alt="插图{i + 1}"><p class="caption">插图 {i + 1}</p></div>')
    else:
        s = (story or "").strip()
        n = len(image_filenames)
        if n == 0:
            first = s[0] if s else ""
            rest = html_module.escape(s[1:]) if len(s) > 1 else ""
            body_parts = [f'<p><span class="snap-dropcap">{first}</span>{rest}</p>']
        else:
            paragraphs = _story_to_paragraphs(s)
            M = len(paragraphs)
            if M == 0:
                paragraphs = [s]
                M = 1
            after_para = [((k + 1) * (M + 1)) // (n + 1) - 1 for k in range(n)]
            body_parts = []
            for i, para in enumerate(paragraphs):
                if not para.strip():
                    continue
                if i == 0:
                    first = para[0] if para else ""
                    rest = html_module.escape(para[1:]) if len(para) > 1 else ""
                    body_parts.append(f'<p><span class="snap-dropcap">{first}</span>{rest}</p>')
                else:
                    body_parts.append(f"<p>{html_module.escape(para)}</p>")
                for k in range(n):
                    if after_para[k] == i and k < len(image_filenames):
                        fn = image_filenames[k]
                        body_parts.append(f'<div class="snap-illus"><img src="{html_module.escape(fn)}" alt="插图{k + 1}"><p class="caption">插图 {k + 1}</p></div>')
    article_body = "\n".join(p for p in body_parts if p)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>Visual Snapshot</title>{css}</head>
<body>
{back_link}
<div class="snap-card-title" style="font-size:1.25rem;font-weight:600;color:#1f2937;margin-bottom:0.25rem;">Visual Snapshot</div>
<div class="snap-card-subtitle" style="font-size:0.8rem;color:#6b7280;margin-bottom:0.75rem;">Story & character illustrations</div>
<article class="snap-article">
{article_body}
</article>
</body>
</html>
"""


def _update_vsnapshot_index() -> None:
    """Scan VSnapshot/snapshot_* dirs and rewrite VSNAPSHOT_DIR/index.html with links (newest first)."""
    VSNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    subdirs = [d for d in VSNAPSHOT_DIR.iterdir() if d.is_dir() and d.name.startswith("snapshot_") and (d / "index.html").exists()]
    subdirs.sort(key=lambda d: d.name, reverse=True)
    lines = [
        "<!DOCTYPE html>",
        "<html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>Visual Snapshots</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:700px;margin:0 auto;padding:2rem;} a{color:#7c3aed;} ul{list-style:none;padding:0;} li{margin:0.5rem 0;}</style>",
        "</head><body>",
        "<h1>Visual Snapshots</h1>",
        "<p>All exported snapshots. Click to open.</p>",
        "<ul>",
    ]
    for d in subdirs:
        ts_display = d.name.replace("snapshot_", "").replace("_", " ", 1)
        lines.append(f'<li><a href="{d.name}/index.html">Snapshot {ts_display}</a></li>')
    lines.append("</ul></body></html>")
    index_path = VSNAPSHOT_DIR / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _export_snap_to_vsnapshot(story: str, gallery_items: list) -> str:
    """Save current Visual Snapshot as HTML + images in VSnapshot/snapshot_YYYY-MM-DD_HHMMSS/; update index. Return message for UI."""
    if not (story or "").strip() and not (gallery_items or []):
        return "无内容可导出（请先生成 Visual Snapshot）。"
    try:
        VSNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        snap_dir = VSNAPSHOT_DIR / f"snapshot_{ts}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        image_filenames = []
        for i, item in enumerate(gallery_items or []):
            if not item or not isinstance(item, (list, tuple)) or len(item) < 1:
                continue
            file_path = item[0] if isinstance(item[0], str) else None
            if file_path and os.path.isfile(file_path):
                ext = Path(file_path).suffix or ".jpg"
                name = f"image_{i + 1}{ext}"
                dest = snap_dir / name
                shutil.copy2(file_path, dest)
                image_filenames.append(name)
        html_content = _build_snapshot_page_html(story or "", image_filenames)
        with open(snap_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        _update_vsnapshot_index()
        return f"已保存至 VSnapshot/snapshot_{ts}/"
    except Exception as e:
        return f"导出失败：{e!s}"



def run_snap(
    snap_gallery_state: list,
    character_prompt: str = "",
    chat_backend: str = "lan",
    xai_api_key: str = "",
    entries: list[dict] | None = None,
    log_path: Path | None = None,
    last_snap_entry_count: int | None = None,
):
    """
    Generator: entries + character_prompt -> story -> N illustration prompts -> N× xAI Imagine (N=SNAP_NUM_IMAGES).
    Yields (gallery_list, status_message, gallery_list, story) after each image so the UI updates one-by-one.
    Pass entries (list of {user, assistant} dicts) or log_path; when using log_path, state is persisted in SNAP_STATE_FILE.
    last_snap_entry_count: when provided (e.g. by Telegram), used for since_snap mode; caller must persist after success.
    """
    state = snap_gallery_state or []
    entries_provided = entries is not None
    state_path = None
    if not entries_provided:
        path = log_path if log_path is not None else CONVERSATIONS_LOG
        state_path = path.parent / "snap_state.json" if path and path.parent else SNAP_STATE_FILE
        if SNAP_STORY_MODE == "since_snap":
            last_snap_entry_count = _load_snap_state(state_path)
            entries = _get_last_n_turns(SNAP_MAX_TURNS_SINCE_SNAP, log_path=log_path)
        else:
            entries = _get_last_n_turns(SNAP_MAX_TURNS, log_path=log_path)
    else:
        entries = list(entries)
    if not entries:
        yield (state, "暂无对话记录，请先进行几轮对话。", state, "")
        return
    story_path = CHAT_LOG_DIR / "story.md"
    previous_story = ""
    if story_path.exists():
        try:
            previous_story = story_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass
    story = _dialog_and_prompt_to_story(
        entries,
        character_prompt,
        chat_backend,
        xai_api_key,
        last_snap_entry_count=last_snap_entry_count,
        previous_story=previous_story,
    )
    if not story:
        yield (state, "无法根据对话生成故事，请检查所选 Chat 后端（LAN MLX 或 Grok）是否可用。", state, "")
        return
    try:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(story_path, "w", encoding="utf-8") as f:
            f.write(story)
        if not entries_provided:
            _save_snap_state(len(entries), state_path)
    except Exception:
        pass
    saved_design = _load_snap_character_design()
    prompts, design_to_use = _story_to_three_image_prompts(story, chat_backend, xai_api_key, saved_character_design=saved_design)
    if not saved_design and design_to_use:
        _save_snap_character_design(design_to_use)
    if len(prompts) < SNAP_NUM_IMAGES:
        yield (state, "无法根据故事生成三张插图提示词，请重试。", state, "")
        return
    api_key = (os.environ.get("XAI_API_KEY") or "").strip()
    if not api_key:
        yield (state, "xAI 图片生成失败：请设置 XAI_API_KEY（.env 或环境变量）。", state, "")
        return
    current = list(state)
    captions = tuple(f"插图{j + 1}" for j in range(SNAP_NUM_IMAGES))
    design_block = (design_to_use or "").strip()
    for i, prompt in enumerate(prompts[:SNAP_NUM_IMAGES]):
        parts = [(prompt or "").strip()]
        # Do not append full design_block to every image (was causing all 4 characters to appear in each image).
        # Scene prompt alone carries who is in frame; style suffix keeps consistency.
        parts.append(SNAP_IMAGE_STYLE)
        full_prompt = " ".join(p for p in parts if p).strip()
        suffix = f"_{i + 1}"
        path = None
        # First attempt
        try:
            path = _call_xai_imagine(full_prompt, api_key, filename_suffix=suffix)
        except requests.RequestException:
            pass
        except Exception:
            pass
        # Retry once on failure (same prompt)
        if not path:
            try:
                path = _call_xai_imagine(full_prompt, api_key, filename_suffix=suffix)
            except requests.RequestException:
                pass
            except Exception:
                pass
        # If still failed, try safety-rewritten prompt
        if not path and (prompt or "").strip():
            rewritten = _rewrite_image_prompt_for_safety(full_prompt, chat_backend, xai_api_key)
            if rewritten:
                retry_prompt = (rewritten.strip() + " " + SNAP_IMAGE_STYLE).strip()
                try:
                    path = _call_xai_imagine(retry_prompt, api_key, filename_suffix=suffix)
                except Exception:
                    pass
                # Retry once more if safety-rewrite attempt failed
                if not path:
                    try:
                        path = _call_xai_imagine(retry_prompt, api_key, filename_suffix=suffix)
                    except Exception:
                        pass
        if path:
            abs_path = os.path.abspath(path)
            current.append((abs_path, captions[i]))
            n = len(current) - len(state)
            status = f"已生成 {n}/{SNAP_NUM_IMAGES} 张插图。" if n < SNAP_NUM_IMAGES else ""
            yield (list(current), status, list(current), story)
    if len(current) == len(state):
        yield (state, "xAI 图片生成失败：未返回图片。", state, story)


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Build a minimal WAV file from raw PCM 16-bit mono little-endian bytes."""
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * (bits_per_sample // 8)
    block_align = n_channels * (bits_per_sample // 8)
    data_size = len(pcm_bytes)
    chunk_size = 4 + 8 + 16 + 8 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm_bytes


def _strip_parenthetical_for_tts(text: str) -> str:
    """
    TTS 前清洗：移除各类括号及其中的内容（含括号本身）。
    括号内多为动作、表情等非对白（如 扑过来抱抱），不送入 TTS，仅对白部分会被朗读。
    支持：英文 ()、中文 （）、方括号【】、［］、「」等。仅用于语音合成。
    """
    if not text:
        return ""
    s = str(text)
    for _ in range(20):
        prev = s
        s = re.sub(r"\([^()]*\)", "", s)
        s = re.sub(r"（[^（）]*）", "", s)
        s = re.sub(r"\[[^\]\[]*\]", "", s)
        s = re.sub(r"【[^】]*】", "", s)
        s = re.sub(r"［[^］]*］", "", s)
        s = re.sub(r"「[^」]*」", "", s)
        s = re.sub(r"〈[^〉]*〉", "", s)
        s = re.sub(r"《[^》]*》", "", s)
        s = re.sub(r"〔[^〕]*〕", "", s)
        s = re.sub(r"﹙[^﹚]*﹚", "", s)
        if s == prev:
            break
    s = re.sub(r"[()（）\[\]【】［］「」〈〉《》〔〕﹙﹚]", "", s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _strip_complete_round_parens(text: str) -> str:
    """Remove complete round-bracket pairs （）() and mixed; leaves unpaired brackets."""
    if not text:
        return ""
    s = str(text)
    for _ in range(20):
        prev = s
        s = re.sub(r"\([^()]*\)", "", s)
        s = re.sub(r"（[^（）]*）", "", s)
        s = re.sub(r"[（(][^）)]*[）)]", "", s)
        if s == prev:
            break
    return s


def _process_segment_for_tts(seg: str, inside_parens: bool) -> tuple[str, bool]:
    """
    Process one TTS segment with cross-segment parenthesis state.
    Case 1: segment has complete () pairs -> strip them, send the rest.
    Case 2: segment has unclosed （ or ( -> send only before first such; set inside_parens.
    When inside_parens: skip until first ） or ), then send content after; clear state.
    Returns (text_to_send, new_inside_parens).
    """
    if not seg:
        return "", inside_parens
    if inside_parens:
        for i, c in enumerate(seg):
            if c in "）)":
                after = _strip_complete_round_parens(seg[i + 1 :])
                idx_open = next((j for j, ch in enumerate(after) if ch in "（("), -1)
                if idx_open >= 0:
                    to_send = after[:idx_open].strip()
                    return (to_send, True) if to_send else ("", True)
                return (after.strip(), False) if after.strip() else ("", False)
        return "", True
    s = _strip_complete_round_parens(seg)
    idx_open = next((j for j, ch in enumerate(s) if ch in "（("), -1)
    if idx_open >= 0:
        to_send = s[:idx_open].strip()
        return (to_send, True) if to_send else ("", True)
    return (s.strip(), False) if s.strip() else ("", False)


def _split_next_tts_segment(
    buffer: str,
    max_segment_chars: int | None = None,
    sentence_max_chars: int | None = None,
) -> tuple[str, str]:
    """
    从 buffer 中取出下一段用于 TTS 的文本（到句号/问号/感叹号/换行或长度上限），
    返回 (segment, remaining)。单段不超过 max_segment_chars（默认 TTS_MAX_SEGMENT_CHARS）。
    """
    max_seg = max_segment_chars if max_segment_chars is not None else TTS_MAX_SEGMENT_CHARS
    sent_max = sentence_max_chars if sentence_max_chars is not None else TTS_SENTENCE_MAX_CHARS
    if not buffer or not buffer.strip():
        return "", ""
    s = buffer.strip()
    if len(s) <= max_seg:
        for sep in ("。", "！", "？", "!", "?", ".", "\n"):
            i = s.rfind(sep)
            if i != -1:
                return s[: i + 1].strip(), s[i + 1 :].strip()
        if len(s) <= sent_max:
            return s, ""
        seg = s[:sent_max]
        return seg, s[sent_max:].strip()
    chunk = s[:max_seg]
    for sep in ("。", "！", "？", "!", "?", ".", "\n"):
        i = chunk.rfind(sep)
        if i != -1:
            return chunk[: i + 1].strip(), s[i + 1 :].strip()
    seg = chunk[:sent_max]
    return seg, s[len(seg) :].strip()


def _play_wav_file(wav_path: str) -> None:
    """Play a WAV file with system player (afplay on macOS, aplay on Linux)."""
    import sys
    try:
        if sys.platform == "darwin":
            subprocess.run(["afplay", wav_path], check=True, timeout=300, capture_output=True)
        elif sys.platform.startswith("linux"):
            subprocess.run(["aplay", "-q", wav_path], check=True, timeout=300, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _dlog("TTS playback failed", {"path": wav_path, "error": str(type(e).__name__), "message": str(e)}, "play_wav")


def _tts_lan_one_segment(text: str, url: str, voice: str = "", instruction_prompt: str = "") -> bool:
    """局域网 TTS：对一段文本请求 WAV，写入临时文件并播放。CustomVoice 支持 extra_body.instruction_prompt。"""
    if not text or not url:
        return False
    text = _strip_parenthetical_for_tts(text)
    if not text:
        return False
    body: dict = {
        "model": LAN_TTS_MODEL,
        "input": text,
        "response_format": "wav",
        "speed": 1.0,
    }
    if (voice or "").strip():
        body["voice"] = voice.strip()
    if (instruction_prompt or "").strip():
        body["extra_body"] = {"instruction_prompt": instruction_prompt.strip()}
    try:
        r = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=60,
        )
        content_type = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        raw = r.content
        _dlog("LAN TTS response", {"url": url, "status": r.status_code, "content_type": content_type, "len": len(raw), "first_bytes": raw[:20].hex() if len(raw) >= 20 else raw.hex()}, "lan_tts")
        r.raise_for_status()
        wav_bytes = raw
        if len(raw) >= 1 and raw[:1] == b"{":
            try:
                body = json.loads(raw.decode("utf-8"))
                b64 = body.get("audio") or body.get("data") or body.get("content")
                if b64 is not None:
                    wav_bytes = base64.b64decode(b64)
                    _dlog("LAN TTS decoded base64", {"decoded_len": len(wav_bytes)}, "lan_tts")
            except Exception as e:
                _dlog("LAN TTS JSON/base64 decode failed", {"error": str(e), "preview": raw[:200]}, "lan_tts")
                return False
        if len(wav_bytes) < 100:
            _dlog("LAN TTS audio too short", {"len": len(wav_bytes)}, "lan_tts")
            return False
        if wav_bytes[:4] != b"RIFF":
            _dlog("LAN TTS not WAV (no RIFF)", {"first4": wav_bytes[:4].hex()}, "lan_tts")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        _play_wav_file(path)
        try:
            os.unlink(path)
        except Exception:
            pass
        return True
    except requests.RequestException as e:
        resp = getattr(e, "response", None)
        _dlog("LAN TTS request failed", {"url": url, "error": str(e), "response_preview": (resp.text[:300] if resp is not None else None)}, "lan_tts")
        return False
    except Exception as e:
        _dlog("LAN TTS error", {"url": url, "error": str(type(e).__name__), "message": str(e)}, "lan_tts")
        return False


def _tts_lan_sender_worker(segment_queue: queue.Queue, lan_url: str, lan_voice: str = "", lan_instruction: str = "") -> None:
    """从 segment_queue 取段，逐段调用局域网 TTS 并播放。"""
    while True:
        try:
            seg = segment_queue.get(timeout=30)
        except queue.Empty:
            continue
        if seg is None:
            break
        cleaned = _strip_parenthetical_for_tts(seg)
        if cleaned:
            _dlog("LAN TTS segment", {"len": len(cleaned), "preview": cleaned[:50]}, "lan_tts")
            _tts_lan_one_segment(cleaned, lan_url, voice=lan_voice, instruction_prompt=lan_instruction)
        else:
            _dlog("LAN TTS segment empty after clean", {"orig_len": len(seg)}, "lan_tts")


def _tts_playback_worker(
    audio_queue: queue.Queue,
    stop_event: threading.Event,
    fallback_wav_path: str | None = None,
) -> None:
    """
    Consume PCM from audio_queue. Prefer pyaudio for real-time playback.
    If pyaudio is unavailable or fails, collect PCM and play via WAV file (afplay/aplay).
    """
    use_pyaudio = False
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_SAMPLE_RATE,
            output=True,
        )
        use_pyaudio = True
    except (ImportError, OSError, Exception):
        pa = None
        stream = None

    collected: list[bytes] = []
    while not stop_event.is_set():
        try:
            data = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if data is None:
            break
        if use_pyaudio and stream is not None:
            try:
                stream.write(data)
            except Exception:
                use_pyaudio = False
                collected.append(data)
        else:
            collected.append(data)

    if pa and stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            pa.terminate()
        except Exception:
            pass

    # Fallback: write collected PCM to WAV and play with system player（不删除，供界面点击播放）
    if collected:
        pcm = b"".join(collected)
        if pcm:
            wav_bytes = _pcm_to_wav(pcm, TTS_SAMPLE_RATE)
            path = fallback_wav_path or tempfile.mktemp(suffix=".wav")
            try:
                with open(path, "wb") as f:
                    f.write(wav_bytes)
                _play_wav_file(path)
            except Exception:
                pass


def _tts_sender_worker(
    api_key: str,
    segment_queue: queue.Queue,
    audio_queue: queue.Queue,
) -> None:
    """Run CosyVoice streaming: consume segments, streaming_call each, then streaming_complete."""
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    try:
        from dashscope.audio.tts_v2 import (
            AudioFormat,
            ResultCallback,
            SpeechSynthesizer,
        )
    except ImportError:
        return
    # Wrap our queue-putting callback so it matches ResultCallback interface
    class Callback(ResultCallback):
        def __init__(self, q: queue.Queue):
            self._q = q

        def on_data(self, data: bytes) -> None:
            if data:
                self._q.put(data)

        def on_open(self) -> None:
            pass

        def on_complete(self) -> None:
            pass

        def on_error(self, message: str) -> None:
            pass

        def on_close(self) -> None:
            pass

        def on_event(self, message: str) -> None:
            pass

    cb = Callback(audio_queue)
    synthesizer = SpeechSynthesizer(
        model="cosyvoice-v3-flash",
        voice="longanhuan",
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=cb,
    )
    while True:
        try:
            seg = segment_queue.get(timeout=30)
        except queue.Empty:
            continue
        if seg is None:
            break
        cleaned = _strip_parenthetical_for_tts(seg)
        if not cleaned:
            continue
        try:
            synthesizer.streaming_call(cleaned)
        except Exception:
            pass
    try:
        synthesizer.streaming_complete()
    except Exception:
        pass
    audio_queue.put(None)


def _tts_sync_one_shot(text: str, api_key: str) -> bool:
    """
    同步合成整段文本为语音，写入 TTS_OUTPUT_WAV。
    流式无数据时用作备用。返回 True 表示成功写入。
    """
    if not text or not text.strip():
        return False
    text = _strip_parenthetical_for_tts(text)
    if not text or len(text) > TTS_MAX_SEGMENT_CHARS:
        text = text[:TTS_MAX_SEGMENT_CHARS] if text else ""
    if not text:
        return False
    if api_key:
        os.environ["DASHSCOPE_API_KEY"] = api_key
    try:
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
    except ImportError:
        return False
    try:
        syn = SpeechSynthesizer(
            model="cosyvoice-v3-flash",
            voice="longanhuan",
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        )
        result = syn.call(text)
        if result is None:
            return False
        if hasattr(result, "get_audio_frame"):
            raw = result.get_audio_frame()
        elif isinstance(result, bytes):
            raw = result
        else:
            raw = getattr(result, "output", None) or getattr(result, "data", None)
            if isinstance(raw, bytes):
                pass
            elif raw is not None and hasattr(raw, "read"):
                raw = raw.read()
            else:
                raw = None
        if not raw or not isinstance(raw, bytes):
            return False
        wav = _pcm_to_wav(raw, TTS_SAMPLE_RATE)
        with open(TTS_OUTPUT_WAV, "wb") as f:
            f.write(wav)
        _play_wav_file(TTS_OUTPUT_WAV)
        return True
    except Exception:
        return False


def _start_tts_session(api_key: str, tts_backend: str = "dashscope", lan_tts_url: str = "", lan_tts_voice: str = "", lan_tts_instruction: str = ""):
    """
    Start TTS for one reply: playback thread + sender thread (or LAN sender only).
    Returns (push_segment_fn, finish_fn). finish_fn() returns path to WAV (or None).
    """
    if tts_backend == "lan" and ((lan_tts_url or "").strip() or LAN_TTS_DEFAULT_URL):
        url = (lan_tts_url or "").strip() or LAN_TTS_DEFAULT_URL
        segment_queue = queue.Queue()
        sender = threading.Thread(
            target=_tts_lan_sender_worker,
            args=(segment_queue, url, (lan_tts_voice or "").strip(), (lan_tts_instruction or "").strip()),
            daemon=True,
        )
        sender.start()

        def push_segment(text: str) -> None:
            if text and text.strip():
                segment_queue.put(text.strip())

        def finish() -> str | None:
            segment_queue.put(None)
            sender.join(timeout=60)
            return None

        return push_segment, finish

    segment_queue = queue.Queue()
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    playback = threading.Thread(
        target=_tts_playback_worker,
        args=(audio_queue, stop_event, TTS_OUTPUT_WAV),
        daemon=True,
    )
    sender = threading.Thread(
        target=_tts_sender_worker,
        args=(api_key, segment_queue, audio_queue),
        daemon=True,
    )
    playback.start()
    sender.start()

    def push_segment(text: str) -> None:
        if text and text.strip():
            segment_queue.put(text.strip())

    def finish() -> str | None:
        segment_queue.put(None)
        sender.join(timeout=60)
        audio_queue.put(None)
        # 等播放线程取完队列并播完再返回，不要先 set stop_event 否则会半路退出
        playback.join(timeout=300)
        stop_event.set()
        if os.path.exists(TTS_OUTPUT_WAV) and os.path.getsize(TTS_OUTPUT_WAV) > 500:
            return TTS_OUTPUT_WAV
        return None

    return push_segment, finish


def _history_to_tuples(history: list) -> list:
    """Convert Gradio 4 message list to (user, assistant) pairs for internal use."""
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
    """Convert (user, assistant) pairs to Gradio 4 message format."""
    out = []
    for user, assistant in history:
        if user:
            out.append({"role": "user", "content": user})
        if assistant:
            out.append({"role": "assistant", "content": assistant})
    return out


def build_messages(character_prompt: str, history: list, user_message: str) -> list:
    """Build API messages: system (character), history, current user message."""
    system = (character_prompt or "").strip() or DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system}]
    tuples = _history_to_tuples(history)
    for user, assistant in tuples:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


TTS_CUMULATIVE_CHAR_LIMIT = 200_000  # CosyVoice streaming session limit


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
    chat_backend: str = "lan",
    xai_api_key: str = "",
):
    """Send request to chat API (LAN MLX or Grok) and stream reply; yield (history, '', tts_status) for Gradio."""
    if not (message or message.strip()):
        yield _tuples_to_messages(_history_to_tuples(history)), "", "", None
        return

    tuples = _history_to_tuples(history)
    new_tuples = tuples + [[message.strip(), ""]]
    yield _tuples_to_messages(new_tuples), "", "", None

    url, headers = _get_chat_config(chat_backend or "lan", xai_api_key or "")
    if (chat_backend or "").strip().lower() == "grok" and not (headers.get("Authorization") or "").replace("Bearer ", "").strip():
        yield _tuples_to_messages(tuples + [[message.strip(), "Grok AI 需要设置 XAI_API_KEY（.env 或环境变量）。"]]), "", "", None
        return

    messages = build_messages(character_prompt, history, message.strip())
    payload = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
    }
    if (chat_backend or "").strip().lower() == "grok":
        payload["model"] = GROK_CHAT_MODEL

    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        is_grok = (chat_backend or "").strip().lower() == "grok"
        if is_grok:
            error_msg = _extract_api_error_message(e, "Grok 请求失败，详见 .cursor/debug.log 中 hypothesisId=grok_err。")
        else:
            error_msg = "Could not reach MLX server. Is it running on port 8000?"
        yield _tuples_to_messages(tuples + [[message.strip(), error_msg]]), "", "", None
        return

    tts_push = None
    tts_finish = None
    tts_buffer = ""
    tts_inside_parens = False
    tts_total_chars = 0
    api_key = (tts_api_key or "").strip() or (os.environ.get("DASHSCOPE_API_KEY") or "").strip() or ""
    use_lan = (tts_backend or "dashscope").strip().lower() == "lan"
    lan_url = (lan_tts_url or "").strip() or LAN_TTS_DEFAULT_URL
    # #region agent log
    _dlog("TTS api_key resolution", {"enable_tts": enable_tts, "tts_backend": tts_backend, "use_lan": use_lan, "has_ui_key": bool((tts_api_key or "").strip()), "env_has_key": bool((os.environ.get("DASHSCOPE_API_KEY") or "").strip()), "api_key_non_empty": bool(api_key)}, "H5")
    # #endregion
    if enable_tts:
        if use_lan:
            try:
                tts_push, tts_finish = _start_tts_session("", tts_backend="lan", lan_tts_url=lan_url, lan_tts_voice=lan_tts_voice, lan_tts_instruction=lan_tts_instruction)
            except Exception:
                tts_push = None
                tts_finish = None
        elif api_key:
            try:
                tts_push, tts_finish = _start_tts_session(api_key, tts_backend="dashscope")
            except Exception:
                tts_push = None
                tts_finish = None

    def _tts_status() -> str:
        if enable_tts and tts_push is None:
            if use_lan:
                return "局域网 TTS 暂时不可用"
            return "TTS 未配置 Key（请设置 DASHSCOPE_API_KEY 或填写 API Key）" if not api_key else "TTS 暂时不可用"
        return "语音播放中…" if tts_push else ""

    full = ""
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload_line = line[6:].decode()
        if payload_line.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload_line)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            part = delta.get("content") or ""
            full += part
            if tts_push is not None:
                tts_buffer += part
                while True:
                    seg, remaining = _split_next_tts_segment(
                        tts_buffer,
                        LAN_TTS_MAX_SEGMENT_CHARS if use_lan else None,
                        LAN_TTS_SENTENCE_MAX_CHARS if use_lan else None,
                    )
                    if not seg:
                        break
                    tts_buffer = remaining
                    to_send, tts_inside_parens = _process_segment_for_tts(seg, tts_inside_parens)
                    if to_send and tts_total_chars <= TTS_CUMULATIVE_CHAR_LIMIT:
                        tts_total_chars += len(to_send)
                        tts_push(to_send)
                status = "语音播放中…"
            else:
                status = _tts_status()
            yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", status, None
        except json.JSONDecodeError:
            continue

    if full.strip():
        _append_conversation_log(character_prompt or "", message.strip(), full)

    if tts_finish is not None:
        last_seg = tts_buffer.strip()
        if last_seg:
            to_send, _ = _process_segment_for_tts(last_seg, tts_inside_parens)
            if to_send and tts_total_chars <= TTS_CUMULATIVE_CHAR_LIMIT:
                tts_push(to_send)
        wav_path = tts_finish()
        if wav_path is None and full.strip() and not use_lan and api_key:
            if _tts_sync_one_shot(full.strip(), api_key):
                wav_path = TTS_OUTPUT_WAV
        yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", "准备就绪。可点击下方播放语音", wav_path
    else:
        yield _tuples_to_messages(tuples + [[message.strip(), full]]), "", "", None


def load_prompt_from_file(file) -> str:
    """Load character prompt from an uploaded file. Return content or empty string."""
    if file is None:
        return ""
    path = file.name if hasattr(file, "name") else file
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def clear_chat():
    """Return empty history for Clear button."""
    return []


def apply_adjustment_to_prompt(current_prompt: str, adjustment_text: str) -> str:
    """Merge adjustment suggestions into system prompt; next chat round will use the updated prompt."""
    current = (current_prompt or "").strip()
    adjustment = (adjustment_text or "").strip()
    if not adjustment:
        return current
    section = "\n\n---\n人设调整建议（下一轮对话中体现）：\n" + adjustment
    return current + section if current else adjustment


SNAP_GALLERY_DBLCLICK_JS = r"""
(function() {
  function attachDblclick() {
    var el = document.querySelector('#snap-gallery');
    if (!el) return false;
    if (el._snapDblclickAttached) return true;
    el._snapDblclickAttached = true;
    el.addEventListener('dblclick', function(ev) {
      var img = ev.target.closest ? ev.target.closest('img') : null;
      if (!img || !img.src) return;
      var url = img.src;
      try {
        if (url.indexOf('http') !== 0) url = new URL(url, window.location.origin).href;
      } catch (_) {}
      if (url) window.open(url, '_blank');
    });
    return true;
  }
  if (!attachDblclick()) {
    var attempts = 0;
    var iv = setInterval(function() {
      if (attachDblclick() || ++attempts > 50) clearInterval(iv);
    }, 200);
  }
})();
"""


def main():
    theme = gr.themes.Soft(primary_hue="violet")
    with gr.Blocks(title="Character Chat", theme=theme, css=UI_CSS, js=SNAP_GALLERY_DBLCLICK_JS, fill_height=True, fill_width=True) as demo:
        # --- Sidebar: persona, chat backend, TTS, creativity, clear ---
        with gr.Sidebar(open=True):
            gr.HTML(
                f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">'
                f'<span style="color:#7c3aed;font-size:1.5rem;">⚡</span>'
                f'<span style="font-size:1.25rem;font-weight:600;color:#374151;font-family:{_FONT_SONGTI_LIGHT};">{APP_HEADER_TITLE}</span>'
                f'</div>'
            )
            character_prompt = gr.Textbox(
                label="Character Persona",
                placeholder="e.g. You are a pirate. Speak in pirate slang. Your name is Red.",
                lines=8,
            )
            file_input = gr.File(label="Load persona file", file_types=[".txt"], elem_classes=["load-persona"])
            file_input.change(load_prompt_from_file, inputs=[file_input], outputs=[character_prompt])
            with gr.Row():
                gr.HTML('<span class="ui-header">Chat 模型</span>')
            chat_backend = gr.Radio(
                choices=[("LAN LLM (MLX)", "lan"), ("Grok AI", "grok")],
                value="lan",
                label="Chat 后端",
                info="Grok 需设置 XAI_API_KEY（与 Snap 共用）",
            )
            with gr.Row():
                enable_tts = gr.Checkbox(label="Voice (TTS)", value=False)
                gr.HTML('<span style="color:#6b7280;font-size:0.875rem;align-self:center;">Streaming TTS</span>')
            with gr.Accordion("TTS 设置", open=False, visible=False) as tts_accordion:
                tts_backend = gr.Radio(
                    choices=[("DashScope CosyVoice", "dashscope"), ("局域网 TTS (MLX)", "lan")],
                    value="lan",
                    label="TTS 后端",
                )
                tts_api_key = gr.Textbox(
                    label="DashScope API Key（仅 CosyVoice 需要）",
                    placeholder="填入阿里云 DashScope API Key；不填则使用 .env 中的 DASHSCOPE_API_KEY",
                    type="password",
                )
                lan_tts_url = gr.Textbox(
                    label="局域网 TTS URL（仅当选择「局域网 TTS」时使用）",
                    placeholder=LAN_TTS_DEFAULT_URL,
                    value=LAN_TTS_DEFAULT_URL,
                )
                lan_tts_voice = gr.Textbox(
                    label="局域网 TTS 音色（voice）",
                    placeholder="可选，如 female/male 或模型文档中的音色名",
                )
                lan_tts_instruction = gr.Textbox(
                    label="局域网 TTS 风格（instruction_prompt）",
                    placeholder="用活泼的年轻女性声音，语气开心带点笑意",
                )
            enable_tts.change(
                lambda on: gr.update(visible=on, open=on),
                inputs=[enable_tts],
                outputs=[tts_accordion],
            )
            creativity_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.7,
                step=0.05,
                label="Creativity (Temp)",
            )
            clear_btn = gr.Button("Clear Chat", variant="secondary")

        # --- Main: header + Tabs (聊天, Snap, 人设优化) ---
        gr.HTML(
            f'<div style="margin-bottom:0.75rem;">'
            f'<div style="display:flex;align-items:center;gap:0.5rem;">'
            f'<span style="width:2rem;height:2rem;border-radius:50%;background:#7c3aed;display:inline-block;"></span>'
            f'<div><div class="ui-header" style="margin:0;">{DISPLAY_MODEL_NAME}</div>'
            f'<div class="ui-subtitle">{DISPLAY_SUBTITLE}</div></div></div></div>'
        )
        with gr.Tab("聊天"):
            chatbot = gr.Chatbot(label="Chat", height="70vh", show_label=False)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Message Nova...",
                    label="Message",
                    show_label=False,
                    scale=10,
                    container=False,
                )
                send_btn = gr.Button("↑", variant="primary", scale=1, min_width=48)

        with gr.Tab("Snap"):
            snap_gallery_state = gr.State([])
            snap_story_state = gr.State("")
            with gr.Row():
                snap_btn = gr.Button("Snap")
            with gr.Column(visible=True, elem_id="visual-snapshot-card"):
                gr.HTML(
                    '<div class="snap-card-title">Visual Snapshot</div>'
                    '<div class="snap-card-subtitle">Story & character illustrations generated from your chat</div>'
                )
                snap_story_display = gr.HTML(
                    value=_build_article_html_with_illustrations("", [], embed_images_as_base64=True),
                    elem_classes=["snap-story-container"],
                )
                snap_gallery = gr.Gallery(
                    label="",
                    show_label=False,
                    columns=1,
                    height=120,
                    object_fit="contain",
                    elem_id="snap-gallery-vertical",
                    visible=False,
                )
                export_archive_btn = gr.Button("Export Archive", variant="secondary")
                snap_export_status = gr.Markdown("", visible=True, elem_id="snap-export-status")
            snap_status = gr.Markdown(visible=False)

        with gr.Tab("人设优化"):
            gr.HTML(
                f'<div style="font-size:1.1rem;font-weight:600;color:#374151;margin-bottom:0.75rem;font-family:{_FONT_SONGTI_LIGHT};">修改建议栏</div>'
            )
            adjustment_direction = gr.Textbox(
                label="",
                lines=6,
                show_label=False,
                interactive=False,
                placeholder="点击下方按钮根据对话记录生成人设调整建议",
            )
            analyze_btn = gr.Button("分析对话并生成调整方向")
            apply_btn = gr.Button("应用调整", variant="secondary")

        def submit(user_msg, hist, prompt, tts_on, tts_key, tts_backend_val, lan_url, lan_voice, lan_instruction, temp, chat_backend_val):
            for h, m, s, audio_path in stream_chat(
                prompt, hist, user_msg,
                enable_tts=tts_on,
                tts_api_key=tts_key or "",
                tts_backend=tts_backend_val or "dashscope",
                lan_tts_url=lan_url or "",
                lan_tts_voice=lan_voice or "",
                lan_tts_instruction=lan_instruction or "",
                temperature=temp,
                chat_backend=chat_backend_val or "lan",
                xai_api_key="",
            ):
                yield h, m

        submit_inputs = [
            msg, chatbot, character_prompt, enable_tts, tts_api_key, tts_backend,
            lan_tts_url, lan_tts_voice, lan_tts_instruction, creativity_slider,
            chat_backend,
        ]
        msg.submit(submit, inputs=submit_inputs, outputs=[chatbot, msg])
        send_btn.click(submit, inputs=submit_inputs, outputs=[chatbot, msg])
        clear_btn.click(clear_chat, outputs=[chatbot])

        def run_analyze(chat_backend_val):
            ideas = analyze_log_for_prompt_ideas(CONVERSATIONS_LOG, chat_backend_val or "lan", "")
            print("调整方向：\n" + ideas)
            return ideas

        analyze_btn.click(run_analyze, inputs=[chat_backend], outputs=[adjustment_direction])

        def on_apply_adjustment(current_prompt, adjustment_text):
            new_prompt = apply_adjustment_to_prompt(current_prompt, adjustment_text)
            return new_prompt, ""  # update character_prompt, clear adjustment_direction

        apply_btn.click(
            on_apply_adjustment,
            inputs=[character_prompt, adjustment_direction],
            outputs=[character_prompt, adjustment_direction],
        )

        def on_snap(state, char_prompt, chat_backend_val):
            for gallery_list, msg, gallery_state, story in run_snap(state, char_prompt or "", chat_backend_val or "lan", ""):
                status_update = gr.update(value=msg, visible=bool(msg.strip())) if msg else gr.update(value="", visible=False)
                story_html = _build_article_html_with_illustrations(story or "", gallery_list or [], embed_images_as_base64=True)
                yield gallery_list, status_update, gallery_state, story_html, (story or "")

        snap_btn.click(
            on_snap,
            inputs=[snap_gallery_state, character_prompt, chat_backend],
            outputs=[snap_gallery, snap_status, snap_gallery_state, snap_story_display, snap_story_state],
        )
        export_archive_btn.click(
            _export_snap_to_vsnapshot,
            inputs=[snap_story_state, snap_gallery_state],
            outputs=[snap_export_status],
        )

    demo.launch()


if __name__ == "__main__":
    main()
