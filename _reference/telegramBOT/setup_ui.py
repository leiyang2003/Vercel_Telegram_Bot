"""
Telegram Agent Setup — simple web UI to create, edit, and run up to 4 Telegram bots.
Config: telegram_bots_config.json. Run: writes .env.bot_N and shows command.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv, dotenv_values
import gradio as gr

# Track running bot processes: agent_id -> Popen
_running_processes: dict[str, subprocess.Popen] = {}

_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR / "telegram_bots_config.json"
MAX_AGENTS = 4
SLOT_IDS = [f"bot_{i}" for i in range(1, MAX_AGENTS + 1)]

# Shared env keys to copy from .env when generating .env.bot_N
SHARED_ENV_KEYS = (
    "XAI_API_KEY",
    "GROK_CHAT_MODEL",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "SKILL_EXEC_ENABLED",
    "MEMORY_SEARCH_ENABLED",
)


def load_config() -> dict:
    """Read telegram_bots_config.json; if missing return default. Optionally seed from .env + workspace."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("agents"), list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    # Default or seed from existing .env and workspace/bot_1
    agents = []
    env_path = _SCRIPT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
        workspace = (os.environ.get("CHATBOT_WORKSPACE") or "workspace/bot_1").strip()
        persona_file = (os.environ.get("TELEGRAM_PERSONA_FILE") or "personas/bot_1/Ani.txt").strip()
        if token or workspace == "workspace/bot_1":
            persona_path = _SCRIPT_DIR / persona_file
            if not persona_path.is_absolute():
                persona_path = _SCRIPT_DIR / persona_file
            persona_text = ""
            if persona_path.exists():
                try:
                    persona_text = persona_path.read_text(encoding="utf-8")
                except Exception:
                    pass
            agents.append({
                "id": "bot_1",
                "name": "QX",
                "enabled": True,
                "telegram_bot_token": token,
                "persona_text": persona_text,
                "persona_filename": Path(persona_file).name if persona_file else "Ani.txt",
                "workspace": workspace,
            })
    return {"agents": agents, "max_agents": MAX_AGENTS}


def save_config(data: dict) -> None:
    """Write telegram_bots_config.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mask_token(token: str) -> str:
    if not token or len(token) < 4:
        return "***"
    return "***" + token[-4:]


def get_agent_by_id(agent_id: str) -> dict | None:
    config = load_config()
    for a in config.get("agents", []):
        if a.get("id") == agent_id:
            return a
    return None


def get_agent_list_for_display() -> list[list]:
    """Return rows for Gradio Dataframe: [id, name, workspace, persona_file, token_masked, enabled]."""
    config = load_config()
    agents_by_id = {a["id"]: a for a in config.get("agents", [])}
    rows = []
    for sid in SLOT_IDS:
        a = agents_by_id.get(sid)
        if a:
            rows.append([
                a.get("id", sid),
                a.get("name", ""),
                a.get("workspace", ""),
                a.get("persona_filename", ""),
                mask_token((a.get("telegram_bot_token") or "")),
                "Yes" if a.get("enabled", True) else "No",
            ])
        else:
            rows.append([sid, "(Not configured)", "", "", "—", "—"])
    return rows


def ensure_workspace_and_persona_dirs(agent_id: str) -> tuple[Path, Path]:
    """Create workspace/<id>/memory, workspace/<id>/MEMORY.md, personas/<id>. Return (workspace_dir, personas_dir)."""
    ws_dir = _SCRIPT_DIR / "workspace" / agent_id
    mem_dir = ws_dir / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_md = ws_dir / "MEMORY.md"
    if not mem_md.exists():
        mem_md.write_text("", encoding="utf-8")
    persona_dir = _SCRIPT_DIR / "personas" / agent_id
    persona_dir.mkdir(parents=True, exist_ok=True)
    return ws_dir, persona_dir


def save_agent(
    slot_id: str,
    name: str,
    telegram_bot_token: str,
    persona_text: str,
    persona_filename: str,
) -> str:
    """Validate, create dirs, write persona file, update config. Return success or error message."""
    name = (name or "").strip()
    token = (telegram_bot_token or "").strip()
    persona_filename = (persona_filename or "persona.txt").strip()
    if not persona_filename.endswith(".txt"):
        persona_filename = persona_filename + ".txt"
    if not name:
        return "Error: Display name is required."
    if not token:
        return "Error: Telegram Bot Token is required (get from @BotFather)."
    if slot_id not in SLOT_IDS:
        return f"Error: Slot must be one of {SLOT_IDS}."

    config = load_config()
    agents = list(config.get("agents", []))
    existing_idx = next((i for i, a in enumerate(agents) if a.get("id") == slot_id), None)

    _, persona_dir = ensure_workspace_and_persona_dirs(slot_id)
    persona_path = persona_dir / persona_filename
    persona_path.write_text((persona_text or "").strip() or "You are a helpful assistant.", encoding="utf-8")

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
    save_config(config)
    return f"Saved agent {name} ({slot_id})."


def run_agent(agent_id: str) -> str:
    """Write .env.bot_<N> and return shell command (fallback for manual run)."""
    env_file = _write_env_for_agent(agent_id)
    if env_file is None:
        return f"Error: Agent {agent_id} not found. Save it first."
    cd = _SCRIPT_DIR
    cmd = f'cd "{cd}" && set -a && source .env.{agent_id} && set +a && python QX.py'
    return f"Wrote {env_file.name}\n\nRun in terminal:\n```\n{cmd}\n```"


def load_agent_into_form(agent_id: str) -> tuple[str, str, str, str, str]:
    """Return (slot_id, name, token_masked_or_empty, persona_text, persona_filename) for form."""
    if not agent_id or agent_id == "Add new":
        return ("bot_1", "", "", "You are a helpful assistant.", "persona.txt")
    agent = get_agent_by_id(agent_id)
    if not agent:
        return (agent_id, "", "", "You are a helpful assistant.", "persona.txt")
    token = agent.get("telegram_bot_token") or ""
    return (
        agent.get("id", agent_id),
        agent.get("name", ""),
        token,  # show full token in form when editing (user can leave as-is or replace)
        agent.get("persona_text", ""),
        agent.get("persona_filename", "persona.txt"),
    )


def _write_env_for_agent(agent_id: str) -> Path | None:
    """Write .env.{agent_id} from config + .env. Return env file path or None if agent not found."""
    agent = get_agent_by_id(agent_id)
    if not agent:
        return None
    load_dotenv(_SCRIPT_DIR / ".env")
    env_lines = []
    for key in SHARED_ENV_KEYS:
        val = os.environ.get(key, "").strip()
        if val:
            env_lines.append(f"{key}={val}")
    env_lines.append(f"TELEGRAM_BOT_TOKEN={agent.get('telegram_bot_token', '')}")
    env_lines.append(f"CHATBOT_WORKSPACE={agent.get('workspace', '')}")
    persona_path = f"personas/{agent_id}/{agent.get('persona_filename', 'persona.txt')}"
    env_lines.append(f"TELEGRAM_PERSONA_FILE={persona_path}")
    env_file = _SCRIPT_DIR / f".env.{agent_id}"
    env_file.write_text("\n".join(env_lines), encoding="utf-8")
    return env_file


def is_bot_running(agent_id: str) -> bool:
    """True if agent_id has a tracked process that is still alive."""
    proc = _running_processes.get(agent_id)
    if proc is None:
        return False
    if proc.poll() is not None:
        _running_processes.pop(agent_id, None)
        return False
    return True


def launch_bot(agent_id: str) -> str:
    """Write .env, start QX.py subprocess, track it. Return status message."""
    if agent_id not in SLOT_IDS:
        return f"Error: Unknown agent {agent_id}."
    if is_bot_running(agent_id):
        return f"{agent_id} is already running."
    env_file = _write_env_for_agent(agent_id)
    if env_file is None:
        return f"Error: Agent {agent_id} not found. Save it first."
    env = {**os.environ, **dotenv_values(env_file)}
    try:
        proc = subprocess.Popen(
            [sys.executable, "QX.py"],
            cwd=str(_SCRIPT_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _running_processes[agent_id] = proc
        return f"Started {agent_id}."
    except Exception as e:
        return f"Error starting {agent_id}: {e}"


def terminate_bot(agent_id: str) -> str:
    """Terminate tracked process for agent_id. Return status message."""
    proc = _running_processes.get(agent_id)
    if proc is None:
        return f"{agent_id} is not running."
    if proc.poll() is not None:
        _running_processes.pop(agent_id, None)
        return f"{agent_id} was already stopped."
    try:
        proc.terminate()
        _running_processes.pop(agent_id, None)
        return f"Stopped {agent_id}."
    except Exception as e:
        return f"Error stopping {agent_id}: {e}"


def get_agent_summary(slot_id: str) -> str:
    """One-line summary: slot | name | workspace | Running|Stopped."""
    agent = get_agent_by_id(slot_id)
    running = is_bot_running(slot_id)
    status = "Running" if running else "Stopped"
    if agent:
        return f"**{slot_id}** | {agent.get('name', '')} | {agent.get('workspace', '')} | {status}"
    return f"**{slot_id}** | (Not configured) | — | {status}"


def get_agents_display_state() -> list[tuple[str, bool, bool]]:
    """For each slot: (summary_md, launch_enabled, terminate_enabled)."""
    result = []
    for sid in SLOT_IDS:
        summary = get_agent_summary(sid)
        configured = get_agent_by_id(sid) is not None
        running = is_bot_running(sid)
        launch_ok = configured and not running
        terminate_ok = running
        result.append((summary, launch_ok, terminate_ok))
    return result


def build_ui():
    with gr.Blocks(title="Telegram Agent Setup", css=".gr-form { margin-bottom: 0.5rem; }") as demo:
        gr.Markdown("# Telegram Agent Setup")
        gr.Markdown("Manage up to 4 Telegram bots. Each has its own workspace (memory, skills) and persona.")
        gr.Markdown(f"Config file: `{CONFIG_PATH.name}` (contains tokens; restrict file access).")

        with gr.Tabs():
            with gr.Tab("Agents"):
                state = get_agents_display_state()
                status_outputs: list[gr.Markdown] = []
                summary_outputs: list[gr.Markdown] = []
                launch_buttons: list[gr.Button] = []
                term_buttons: list[gr.Button] = []

                for i, slot_id in enumerate(SLOT_IDS):
                    summary_txt, launch_ok, term_ok = state[i]
                    with gr.Row():
                        summary_md = gr.Markdown(summary_txt, elem_id=f"summary_{slot_id}")
                        launch_btn = gr.Button(
                            "Launch", variant="primary", elem_id=f"launch_{slot_id}", interactive=launch_ok
                        )
                        term_btn = gr.Button("Terminate", elem_id=f"term_{slot_id}", interactive=term_ok)
                        row_status = gr.Markdown("", elem_id=f"status_{slot_id}")
                    summary_outputs.append(summary_md)
                    launch_buttons.append(launch_btn)
                    term_buttons.append(term_btn)
                    status_outputs.append(row_status)

                all_outputs = status_outputs + summary_outputs + launch_buttons + term_buttons

                def refresh_ui():
                    st = get_agents_display_state()
                    out = [gr.update() for _ in range(4)]  # status unchanged
                    for i in range(4):
                        out.append(st[i][0])
                    for i in range(4):
                        out.append(gr.update(interactive=st[i][1]))
                    for i in range(4):
                        out.append(gr.update(interactive=st[i][2]))
                    return out

                def make_launch_handler(slot_idx: int):
                    def handler():
                        msg = launch_bot(SLOT_IDS[slot_idx])
                        st = get_agents_display_state()
                        out = [msg if i == slot_idx else gr.update() for i in range(4)]
                        for i in range(4):
                            out.append(st[i][0])
                        for i in range(4):
                            out.append(gr.update(interactive=st[i][1]))
                        for i in range(4):
                            out.append(gr.update(interactive=st[i][2]))
                        return out
                    return handler

                def make_terminate_handler(slot_idx: int):
                    def handler():
                        msg = terminate_bot(SLOT_IDS[slot_idx])
                        st = get_agents_display_state()
                        out = [msg if i == slot_idx else gr.update() for i in range(4)]
                        for i in range(4):
                            out.append(st[i][0])
                        for i in range(4):
                            out.append(gr.update(interactive=st[i][1]))
                        for i in range(4):
                            out.append(gr.update(interactive=st[i][2]))
                        return out
                    return handler

                refresh_btn = gr.Button("Refresh")
                refresh_btn.click(fn=refresh_ui, outputs=all_outputs)

                for idx in range(4):
                    launch_buttons[idx].click(
                        fn=make_launch_handler(idx),
                        outputs=all_outputs,
                    )
                    term_buttons[idx].click(
                        fn=make_terminate_handler(idx),
                        outputs=all_outputs,
                    )

                with gr.Row():
                    slot_dropdown = gr.Dropdown(
                        choices=["Add new"] + SLOT_IDS,
                        value="Add new",
                        label="Select agent to edit or run",
                    )
                with gr.Row():
                    edit_btn = gr.Button("Edit selected")
                    run_btn = gr.Button("Run selected (write .env & show command)")
                    run_output = gr.Markdown("")

                def on_run(slot):
                    if not slot or slot == "Add new":
                        return "Select an existing agent (e.g. bot_1) then click Run."
                    return run_agent(slot)

                run_btn.click(fn=on_run, inputs=[slot_dropdown], outputs=[run_output])

            with gr.Tab("Add / Edit agent"):
                with gr.Row():
                    form_slot = gr.Dropdown(choices=SLOT_IDS, value="bot_1", label="Slot")
                    form_name = gr.Textbox(label="Display name", placeholder="e.g. QX, Support Bot")
                form_token = gr.Textbox(
                    label="Telegram Bot Token",
                    type="password",
                    placeholder="From @BotFather",
                )
                form_persona = gr.Textbox(
                    label="Persona (system prompt)",
                    lines=8,
                    placeholder="You are a helpful assistant...",
                )
                form_persona_file = gr.File(
                    label="Or upload persona .txt (loads into text above)",
                    file_types=[".txt"],
                )
                form_persona_filename = gr.Textbox(
                    label="Persona filename (saved under personas/<slot>/)",
                    value="persona.txt",
                    placeholder="Ani.txt",
                )
                save_btn = gr.Button("Save agent", variant="primary")
                save_status = gr.Markdown("")

                def load_form(slot_choice):
                    return load_agent_into_form(slot_choice)

                def on_save(slot, name, token, persona, persona_filename):
                    return save_agent(slot, name, token, persona, persona_filename)

                def persona_file_upload(file):
                    if file is None:
                        return gr.update()
                    try:
                        if isinstance(file, list) and file:
                            path = file[0]
                        elif hasattr(file, "name"):
                            path = file.name
                        else:
                            path = file
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()
                        return gr.update(value=text)
                    except Exception:
                        return gr.update()

                slot_dropdown.change(
                    fn=load_form,
                    inputs=[slot_dropdown],
                    outputs=[form_slot, form_name, form_token, form_persona, form_persona_filename],
                )
                edit_btn.click(
                    fn=load_form,
                    inputs=[slot_dropdown],
                    outputs=[form_slot, form_name, form_token, form_persona, form_persona_filename],
                )
                form_persona_file.upload(fn=persona_file_upload, inputs=[form_persona_file], outputs=[form_persona])
                save_btn.click(
                    fn=on_save,
                    inputs=[form_slot, form_name, form_token, form_persona, form_persona_filename],
                    outputs=[save_status],
                )
                save_btn.click(fn=refresh_ui, outputs=all_outputs)

    return demo


def main():
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
