"""
Bot 配置服务：按 user_id 隔离。
存储支持本地 users/ 与 Vercel Blob（BLOB_READ_WRITE_TOKEN 存在时）。
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from storage import (
    storage_read_json,
    storage_write_json,
    storage_write_text,
    get_user_dir,
    ensure_user_dirs,
)

ROOT = Path(__file__).resolve().parent
MAX_AGENTS = 4
SLOT_IDS = [f"bot_{i}" for i in range(1, MAX_AGENTS + 1)]

SHARED_ENV_KEYS = (
    "XAI_API_KEY",
    "GROK_CHAT_MODEL",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "SKILL_EXEC_ENABLED",
    "MEMORY_SEARCH_ENABLED",
)

CONFIG_REL = "telegram_bots_config.json"


def _user_dir(user_id: str) -> Path:
    return get_user_dir(user_id)


def load_config(user_id: str) -> dict:
    """读取用户 config"""
    data = storage_read_json(user_id, CONFIG_REL, {"agents": [], "max_agents": MAX_AGENTS})
    if isinstance(data.get("agents"), list):
        return data
    return {"agents": [], "max_agents": MAX_AGENTS}


def save_config(user_id: str, data: dict) -> None:
    """保存用户 config"""
    storage_write_json(user_id, CONFIG_REL, data)


def mask_token(token: str) -> str:
    if not token or len(token) < 4:
        return "***"
    return "***" + token[-4:]


def get_agent_by_id(user_id: str, agent_id: str) -> dict | None:
    config = load_config(user_id)
    for a in config.get("agents", []):
        if a.get("id") == agent_id:
            return a
    return None


def _persona_rel(slot_id: str, filename: str) -> str:
    return f"personas/{slot_id}/{filename}"


def ensure_workspace_and_persona_dirs(user_id: str, agent_id: str) -> None:
    """确保 workspace、personas 就绪（本地模式创建目录，Blob 模式无需操作）"""
    ensure_user_dirs(user_id, agent_id)


def save_agent(
    user_id: str,
    slot_id: str,
    name: str,
    telegram_bot_token: str,
    persona_text: str,
    persona_filename: str,
) -> tuple[bool, str]:
    """
    保存 agent。Returns (ok, message).
    """
    name = (name or "").strip()
    token = (telegram_bot_token or "").strip()
    persona_filename = (persona_filename or "persona.txt").strip()
    if not persona_filename.endswith(".txt"):
        persona_filename = persona_filename + ".txt"
    if not name:
        return False, "Display name is required."
    if not token:
        return False, "Telegram Bot Token is required (get from @BotFather)."
    if slot_id not in SLOT_IDS:
        return False, f"Slot must be one of {SLOT_IDS}."

    config = load_config(user_id)
    agents = list(config.get("agents", []))
    existing_idx = next((i for i, a in enumerate(agents) if a.get("id") == slot_id), None)

    ensure_workspace_and_persona_dirs(user_id, slot_id)
    storage_write_text(
        user_id,
        _persona_rel(slot_id, persona_filename),
        (persona_text or "").strip() or "You are a helpful assistant.",
    )

    agent = {
        "id": slot_id,
        "name": name,
        "enabled": True,
        "telegram_bot_token": token,
        "persona_text": (persona_text or "").strip(),
        "persona_filename": persona_filename,
        "workspace": f"workspace/{slot_id}",
    }
    if existing_idx is not None:
        agents[existing_idx] = agent
    else:
        agents.append(agent)
    config["agents"] = agents
    save_config(user_id, config)
    return True, f"Saved {name} ({slot_id})."


def get_agents_for_display(user_id: str) -> list[dict]:
    """返回供前端展示的 agents 列表"""
    config = load_config(user_id)
    agents_by_id = {a["id"]: a for a in config.get("agents", [])}
    out = []
    for sid in SLOT_IDS:
        a = agents_by_id.get(sid)
        if a:
            out.append({
                "id": sid,
                "name": a.get("name", ""),
                "workspace": a.get("workspace", ""),
                "persona_filename": a.get("persona_filename", ""),
                "token_masked": mask_token(a.get("telegram_bot_token") or ""),
                "enabled": a.get("enabled", True),
                "configured": True,
            })
        else:
            out.append({
                "id": sid,
                "name": "",
                "workspace": "",
                "persona_filename": "",
                "token_masked": "—",
                "enabled": False,
                "configured": False,
            })
    return out


def get_agent_detail(user_id: str, slot_id: str) -> dict | None:
    """返回单 agent 详情（含 persona_text），用于编辑"""
    agent = get_agent_by_id(user_id, slot_id)
    if not agent:
        return None
    return {
        "id": agent.get("id"),
        "name": agent.get("name", ""),
        "telegram_bot_token": agent.get("telegram_bot_token", ""),
        "persona_text": agent.get("persona_text", ""),
        "persona_filename": agent.get("persona_filename", "persona.txt"),
        "workspace": agent.get("workspace", ""),
    }


def export_run_command(user_id: str, slot_id: str) -> tuple[bool, str]:
    """
    生成「本地运行」命令。本地模式给出具体路径；Vercel/Blob 模式给出 persona 内联说明。
    """
    agent = get_agent_by_id(user_id, slot_id)
    if not agent:
        return False, f"Agent {slot_id} not found. Save it first."

    load_dotenv(ROOT / ".env")
    env_lines = []
    for key in SHARED_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            env_lines.append(f"{key}={val}")
    env_lines.append(f"TELEGRAM_BOT_TOKEN={agent.get('telegram_bot_token', '')}")

    ud = _user_dir(user_id)
    workspace_abs = str((ud / agent.get("workspace", f"workspace/{slot_id}")).resolve())
    persona_path = ud / "personas" / slot_id / agent.get("persona_filename", "persona.txt")
    chat_log_dir = ud / "chat_logs" / slot_id
    try:
        chat_log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # Vercel 上 users/ 不可写，仅生成说明

    env_lines.append(f"CHATBOT_WORKSPACE={workspace_abs}")
    env_lines.append(f"TELEGRAM_PERSONA_FILE={persona_path}")
    env_lines.append(f"CHAT_LOG_DIR={chat_log_dir.resolve()}")

    persona_text = agent.get("persona_text", "")
    env_block = "\n".join(env_lines)
    cmd = (
        f"# 在 telegramBOT 目录运行:\n"
        f"# cd ../telegramBOT\n"
        f"# 将下方 env 导出后: python QX.py\n\n"
        f"{env_block}\n\n"
        f"# Persona 文件需存在，可将下方内容保存为 personas/{slot_id}/{agent.get('persona_filename', 'persona.txt')}\n"
        f"# ---\n{persona_text[:500]}{'...' if len(persona_text) > 500 else ''}\n# ---"
    )
    return True, cmd


def get_run_package(user_id: str, slot_id: str) -> dict | None:
    """
    返回完整的运行配置包（用于下载）：config、persona 文本、env 模板。
    便于 Blob 用户下载后本地部署。
    """
    agent = get_agent_by_id(user_id, slot_id)
    if not agent:
        return None
    load_dotenv(ROOT / ".env")
    env = {}
    for key in SHARED_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val
    env["TELEGRAM_BOT_TOKEN"] = agent.get("telegram_bot_token", "")
    env["CHATBOT_WORKSPACE"] = f"<本地路径>/workspace/{slot_id}"
    env["TELEGRAM_PERSONA_FILE"] = f"<本地路径>/personas/{slot_id}/{agent.get('persona_filename', 'persona.txt')}"
    env["CHAT_LOG_DIR"] = f"<本地路径>/chat_logs/{slot_id}"
    return {
        "slot_id": slot_id,
        "name": agent.get("name", ""),
        "persona_filename": agent.get("persona_filename", "persona.txt"),
        "persona_text": agent.get("persona_text", ""),
        "env_template": env,
        "readme": (
            "1. 在 telegramBOT 同级创建目录结构: workspace/{slot_id}, personas/{slot_id}, chat_logs/{slot_id}\n"
            f"2. 将 persona_text 保存为 personas/{slot_id}/{agent.get('persona_filename', 'persona.txt')}\n"
            "3. 将 env_template 导出为环境变量，CHATBOT_WORKSPACE/TELEGRAM_PERSONA_FILE/CHAT_LOG_DIR 替换为实际绝对路径\n"
            "4. cd telegramBOT && python QX.py"
        ),
    }
