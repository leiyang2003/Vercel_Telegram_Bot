"""
统一网页应用：Flask 静态 + /reports/ + 温故知新 API。
运行：python app.py，在浏览器打开 http://127.0.0.1:5003/ 或 http://localhost:5003/ 进入首页。
支持 Google 登录；温故知新数据按用户存储。
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

from authlib.integrations.flask_client import OAuth
from flask import Flask, jsonify, redirect, request, send_from_directory, session

from config import (
    DIGEST_TIMEZONE,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    OPENAI_API_KEY,
    PROXIES,
    SECRET_KEY,
)

# 项目根目录、reports 与 static
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
STATIC = ROOT / "static"
USERS_DIR = REPORTS / "users"

USER_SETTINGS = REPORTS / "user_settings.json"


def _default_user_settings() -> dict:
    return {
        "modules": {
            "digest": True,
            "podcast": True,
            "sync_reader": True,
            "japanese_points": True,
        },
        "tts_sync_chunking": "atomic",
    }


def _validate_modules(modules: dict) -> tuple[bool, str]:
    """Validate module toggle dependencies. Returns (ok, message)."""
    if not isinstance(modules, dict):
        return False, "modules must be an object"
    digest_on = bool(modules.get("digest"))
    podcast_on = bool(modules.get("podcast"))
    sync_on = bool(modules.get("sync_reader"))
    jp_on = bool(modules.get("japanese_points"))
    if podcast_on and not digest_on:
        return False, "不能保存：开启 Podcast 时必须开启「ニュース概要（Digest）」"
    if sync_on and not (digest_on and podcast_on):
        return False, "不能保存：开启 同期朗読 时必须同时开启 Digest 与 Podcast"
    if jp_on and not podcast_on:
        return False, "不能保存：开启 Japanese points 时必须开启 Podcast"
    return True, ""

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
app.secret_key = SECRET_KEY
# Ensure session cookie is sent when browser returns from Google (same host as redirect_uri)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # True only over HTTPS

# Google OAuth（Authlib）
oauth = OAuth(app)
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        client_kwargs={"scope": "openid email profile"},
    )


def _safe_user_id(sub: str) -> str:
    """Stable, filesystem-safe user id from Google sub."""
    if not sub:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(sub)).strip() or "unknown"


def _current_user_id():
    """Return session user_id (for storage) or None if not logged in."""
    return session.get("user_id")


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.route("/digest")
@app.route("/digest.html")
def digest():
    return send_from_directory(ROOT, "digest.html")


@app.route("/podcast")
@app.route("/podcast.html")
def podcast():
    return send_from_directory(ROOT, "podcast.html")


@app.route("/sync_reader")
@app.route("/sync_reader.html")
def sync_reader():
    return send_from_directory(ROOT, "sync_reader.html")


@app.route("/japanese_points")
@app.route("/japanese_points.html")
def japanese_points():
    return send_from_directory(ROOT, "japanese_points.html")


@app.route("/settings")
@app.route("/settings.html")
def settings():
    if not _current_user_id():
        return redirect("/login?next=/settings")
    return send_from_directory(ROOT, "settings.html")


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/static/<path:path>")
def serve_static(path: str):
    """提供 static 目录下的 CSS 等静态文件（如 base.css）。"""
    return send_from_directory(STATIC, path)


def _is_user_overridable_report_path(path: str) -> bool:
    """Path 为单层且属于可被用户目录覆盖的白名单。"""
    if ".." in path or "users" in path or "/" in path:
        return False
    return (
        (path.startswith("daily_digest_") and path.endswith(".json"))
        or (path.startswith("podcast_script_") and path.endswith(".md"))
        or (path.startswith("podcast_") and path.endswith(".mp3"))
        or (path.startswith("podcast_") and path.endswith("_sync.json"))
        or (path.startswith("japanese_points_") and path.endswith(".json"))
    )


@app.route("/reports/<path:path>")
def serve_reports(path: str):
    """提供 reports 目录下的 JSON 等静态文件。已登录且用户目录有同名校验通过的文件时优先从用户目录提供。"""
    user_id = _current_user_id()
    if user_id and _is_user_overridable_report_path(path):
        user_path = USERS_DIR / user_id / path
        if user_path.exists():
            return send_from_directory(USERS_DIR / user_id, path)
    return send_from_directory(REPORTS, path)


def _yesterday_yyyymmdd():
    """按 DIGEST_TIMEZONE 计算「昨天」的日期，返回 YYYY-MM-DD。"""
    tz = ZoneInfo(DIGEST_TIMEZONE)
    now = datetime.now(tz)
    yesterday = (now - timedelta(days=1)).date()
    return yesterday.strftime("%Y-%m-%d")


def _find_latest_japanese_points_date(directory: Path) -> Optional[str]:
    """在目录中查找 japanese_points_YYYY-MM-DD.json，返回最新的日期字符串，无则返回 None。"""
    if not directory.is_dir():
        return None
    latest = None
    pattern = re.compile(r"^japanese_points_(\d{4}-\d{2}-\d{2})\.json$")
    for p in directory.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                d = m.group(1)
                if latest is None or d > latest:
                    latest = d
    return latest


def _find_latest_digest_date(directory: Path) -> Optional[str]:
    """在目录中查找 daily_digest_YYYY_MM_DD.json，返回最新的日期字符串 YYYY-MM-DD，无则返回 None。"""
    if not directory.is_dir():
        return None
    latest = None
    pattern = re.compile(r"^daily_digest_(\d{4})_(\d{2})_(\d{2})\.json$")
    for p in directory.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                if latest is None or d > latest:
                    latest = d
    return latest


def _find_latest_podcast_script_date(directory: Path) -> Optional[str]:
    """在目录中查找 podcast_script_YYYY-MM-DD.md，返回最新的日期字符串，无则返回 None。"""
    if not directory.is_dir():
        return None
    latest = None
    pattern = re.compile(r"^podcast_script_(\d{4}-\d{2}-\d{2})\.md$")
    for p in directory.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                d = m.group(1)
                if latest is None or d > latest:
                    latest = d
    return latest


def _find_latest_sync_date(directory: Path) -> Optional[str]:
    """在目录中查找 podcast_YYYY-MM-DD.mp3 且同日的 _sync.json 存在，返回最新日期，无则返回 None。"""
    if not directory.is_dir():
        return None
    latest = None
    pattern = re.compile(r"^podcast_(\d{4}-\d{2}-\d{2})\.mp3$")
    for p in directory.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                date_str = m.group(1)
                sync_path = directory / f"podcast_{date_str}_sync.json"
                if sync_path.exists() and (latest is None or date_str > latest):
                    latest = date_str
    return latest


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """GET：返回当前登录用户设定。POST：保存设定（含模块开关）。需登录。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    path = REPORTS / f"user_settings_{user_id}.json"
    default = _default_user_settings()
    if request.method == "GET":
        data = _read_json(path, default)
        # Merge defaults to be forward-compatible when new keys are added
        merged = default
        if isinstance(data, dict):
            merged = {**default, **data}
            if isinstance(data.get("modules"), dict):
                merged["modules"] = {**default["modules"], **data["modules"]}
        return jsonify(merged)
    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be JSON object"}), 400
    incoming_modules = data.get("modules")
    incoming_tts_chunking = data.get("tts_sync_chunking")
    # If modules omitted, keep existing (or default) modules
    current = _read_json(path, default)
    current_modules = default["modules"]
    if isinstance(current, dict) and isinstance(current.get("modules"), dict):
        current_modules = {**default["modules"], **current["modules"]}
    modules = current_modules
    if incoming_modules is not None:
        if not isinstance(incoming_modules, dict):
            return jsonify({"error": "invalid_modules", "message": "modules must be an object"}), 400
        modules = {
            "digest": bool(incoming_modules.get("digest")),
            "podcast": bool(incoming_modules.get("podcast")),
            "sync_reader": bool(incoming_modules.get("sync_reader")),
            "japanese_points": bool(incoming_modules.get("japanese_points")),
        }
    ok, msg = _validate_modules(modules)
    if not ok:
        return jsonify({"error": "invalid_modules", "message": msg}), 400
    # tts sync chunking (atomic|sentence)
    current_chunking = (current.get("tts_sync_chunking") if isinstance(current, dict) else None) or default["tts_sync_chunking"]
    tts_chunking = current_chunking
    if incoming_tts_chunking is not None:
        if incoming_tts_chunking not in ("atomic", "sentence"):
            return jsonify({"error": "invalid_tts_sync_chunking", "message": "tts_sync_chunking must be 'atomic' or 'sentence'"}), 400
        tts_chunking = incoming_tts_chunking

    out = {"modules": modules, "tts_sync_chunking": tts_chunking}
    _write_json(path, out)
    return jsonify(out)


@app.route("/login")
def login():
    """Redirect to Google OAuth authorization."""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({"error": "Google OAuth not configured"}), 503
    session["next_after_login"] = request.args.get("next", "/")
    # Build redirect_uri from current request host so the callback hits the same host and the
    # session cookie (set on this host) is sent. Otherwise e.g. login on 127.0.0.1 and
    # callback on localhost yields empty session and "mismatching_state".
    # Add the exact URI(s) you use in Google Cloud Console (e.g. both 127.0.0.1 and localhost if needed).
    redirect_uri = request.url_root.rstrip("/") + "/auth/callback"
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/callback")
def auth_callback():
    """Exchange code for tokens, store user in session, redirect to next or /."""
    try:
        token = oauth.google.authorize_access_token()
    except Exception as e:
        return jsonify({"error": "auth_failed", "message": str(e)}), 400
    userinfo = token.get("userinfo") or {}
    sub = userinfo.get("sub")
    if not sub:
        return jsonify({"error": "no user info"}), 400
    user_id = _safe_user_id(sub)
    session["user_id"] = user_id
    session["email"] = userinfo.get("email") or ""
    session["name"] = userinfo.get("name") or userinfo.get("email") or ""
    next_url = session.pop("next_after_login", "/")
    return redirect(next_url)


@app.route("/logout")
def logout():
    """Clear session and redirect to home."""
    session.clear()
    return redirect("/")


@app.route("/api/me")
def api_me():
    """Return current user for frontend: { logged_in, email?, name? }."""
    uid = _current_user_id()
    if not uid:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "email": session.get("email") or "",
        "name": session.get("name") or "",
    })


def _extract_themes_from_text(raw_text: str) -> list[str]:
    """Use OpenAI Chat to extract 3-4 concrete themes (short Chinese labels) from user text. Returns list of up to 4 strings."""
    if not OPENAI_API_KEY:
        raise ValueError("未设置 OPENAI_API_KEY")
    system = (
        "你是一个助手。用户会输入一段描述自己感兴趣方向的文字。"
        "请从中抽取 3～4 个具体的、可作为新闻/资讯分类的主题词（中文短词，如：动漫、游戏、科技、音乐、体育）。"
        "只输出一个 JSON 数组，不要其他说明。例如：[\"动漫\",\"游戏\",\"科技\"]"
    )
    user = raw_text.strip() or "动漫、游戏、科技"
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        timeout=60,
        proxies=PROXIES,
    )
    resp.raise_for_status()
    text = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
    text = text.strip()
    # Parse JSON array: allow ```json ... ``` or raw [...]
    for start in ("[", "```"):
        if start in text:
            idx = text.find(start)
            if start == "[":
                chunk = text[idx:]
            else:
                chunk = text[idx:].replace("```json", "").replace("```", "").strip()
                if chunk.startswith("["):
                    pass
                else:
                    continue
            end = chunk.rfind("]")
            if end != -1:
                try:
                    arr = json.loads(chunk[: end + 1])
                    if isinstance(arr, list):
                        themes = [str(x).strip()[: 20] for x in arr if str(x).strip()][:4]
                        return themes
                except json.JSONDecodeError:
                    pass
    return []


@app.route("/api/user/interests", methods=["GET", "POST"])
def api_user_interests():
    """GET：返回当前用户的兴趣 raw + themes（需登录）。POST：body { raw_text }，抽取 3～4 个主题并保存。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    path = REPORTS / f"user_interests_{user_id}.json"
    default = {"raw": "", "themes": []}
    if request.method == "GET":
        data = _read_json(path, default)
        return jsonify({"raw": data.get("raw", ""), "themes": data.get("themes", [])})
    data = request.get_json(force=True, silent=True) or {}
    raw_text = data.get("raw_text", "")
    raw_text = raw_text.strip() if isinstance(raw_text, str) else ""
    try:
        themes = _extract_themes_from_text(raw_text) if raw_text else []
    except Exception as e:
        return jsonify({"error": "extract_failed", "message": str(e)}), 500
    out = {"raw": raw_text, "themes": themes}
    _write_json(path, out)
    return jsonify({"raw": raw_text, "themes": themes})


@app.route("/api/japanese_points/latest")
def api_japanese_points_latest():
    """返回最新存在的 japanese_points_YYYY-MM-DD.json 的完整内容（单词与文法）。已登录时优先读用户目录。"""
    user_id = _current_user_id()
    candidates = []
    if user_id:
        user_dir = USERS_DIR / user_id
        d = _find_latest_japanese_points_date(user_dir)
        if d:
            candidates.append(d)
    d_rep = _find_latest_japanese_points_date(REPORTS)
    if d_rep:
        candidates.append(d_rep)
    if not candidates:
        return jsonify({"error": "no japanese_points report found (run japanese_points.py to generate)"}), 404
    date_str = max(candidates)
    if user_id:
        path = USERS_DIR / user_id / f"japanese_points_{date_str}.json"
        if not path.exists():
            path = REPORTS / f"japanese_points_{date_str}.json"
    else:
        path = REPORTS / f"japanese_points_{date_str}.json"
    if not path.exists():
        return jsonify({"error": f"no japanese_points report for {date_str}"}), 404
    data = _read_json(path, None)
    if data is None:
        return jsonify({"error": "failed to read report"}), 500
    if "report_date" not in data:
        data["report_date"] = date_str
    words = data.get("words") or []
    grammar = data.get("grammar") or []
    print(f"[japanese_points] 最新を読み込む: japanese_points_{date_str}.json を表示中。語彙 {len(words)} 件、文法 {len(grammar)} 件。")
    return jsonify(data)


@app.route("/api/digest/latest")
def api_digest_latest():
    """返回最新存在的 daily_digest JSON 的完整内容。已登录时优先读用户目录。"""
    user_id = _current_user_id()
    candidates = []
    if user_id:
        d = _find_latest_digest_date(USERS_DIR / user_id)
        if d:
            candidates.append(d)
    d_rep = _find_latest_digest_date(REPORTS)
    if d_rep:
        candidates.append(d_rep)
    if not candidates:
        return jsonify({"error": "no digest report found (run digest.py to generate)"}), 404
    date_str = max(candidates)
    file_date = date_str.replace("-", "_")
    if user_id:
        path = USERS_DIR / user_id / f"daily_digest_{file_date}.json"
        if not path.exists():
            path = REPORTS / f"daily_digest_{file_date}.json"
    else:
        path = REPORTS / f"daily_digest_{file_date}.json"
    if not path.exists():
        return jsonify({"error": f"no digest report for {date_str}"}), 404
    data = _read_json(path, None)
    if data is None:
        return jsonify({"error": "failed to read report"}), 500
    if "report_date" not in data:
        data["report_date"] = date_str
    return jsonify(data)


@app.route("/api/podcast/latest")
def api_podcast_latest():
    """返回最新存在的 podcast_script 的日期与内容。已登录时优先读用户目录。"""
    user_id = _current_user_id()
    candidates = []
    if user_id:
        d = _find_latest_podcast_script_date(USERS_DIR / user_id)
        if d:
            candidates.append(d)
    d_rep = _find_latest_podcast_script_date(REPORTS)
    if d_rep:
        candidates.append(d_rep)
    if not candidates:
        return jsonify({"error": "no podcast script found (run podcast.py to generate)"}), 404
    date_str = max(candidates)
    if user_id:
        path = USERS_DIR / user_id / f"podcast_script_{date_str}.md"
        if not path.exists():
            path = REPORTS / f"podcast_script_{date_str}.md"
    else:
        path = REPORTS / f"podcast_script_{date_str}.md"
    if not path.exists():
        return jsonify({"error": f"no podcast script for {date_str}"}), 404
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return jsonify({"error": "failed to read script"}), 500
    return jsonify({"date": date_str, "content": content})


@app.route("/api/sync_reader/latest")
def api_sync_reader_latest():
    """返回最新存在的同步朗读日期（需同时有 .mp3 与 _sync.json）。已登录时优先读用户目录。"""
    user_id = _current_user_id()
    candidates = []
    if user_id:
        d = _find_latest_sync_date(USERS_DIR / user_id)
        if d:
            candidates.append(d)
    d_rep = _find_latest_sync_date(REPORTS)
    if d_rep:
        candidates.append(d_rep)
    if not candidates:
        return jsonify({"error": "no sync report found (run tts.py --sync to generate)"}), 404
    date_str = max(candidates)
    return jsonify({"date": date_str})


@app.route("/api/review/words", methods=["GET", "POST"])
def api_review_words():
    """GET：返回当前用户的温故知新-单词列表。POST：body 为完整数组，覆盖写入。需登录。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    path = REPORTS / f"review_words_{user_id}.json"
    if request.method == "GET":
        data = _read_json(path, [])
        return jsonify(data)
    data = request.get_json(force=True, silent=True) or []
    if not isinstance(data, list):
        return jsonify({"error": "body must be JSON array"}), 400
    _write_json(path, data)
    return jsonify(data)


@app.route("/api/review/grammar", methods=["GET", "POST"])
def api_review_grammar():
    """GET：返回当前用户的温故知新-文法列表。POST：body 为完整数组，覆盖写入。需登录。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    path = REPORTS / f"review_grammar_{user_id}.json"
    if request.method == "GET":
        data = _read_json(path, [])
        return jsonify(data)
    data = request.get_json(force=True, silent=True) or []
    if not isinstance(data, list):
        return jsonify({"error": "body must be JSON array"}), 400
    _write_json(path, data)
    return jsonify(data)


@app.route("/api/review/progress", methods=["GET", "POST"])
def api_review_progress():
    """GET：返回当前用户进度 { mode, index }。POST：body { mode, index } 保存。需登录。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    path = REPORTS / f"review_progress_{user_id}.json"
    default = {"mode": "words", "index": 0}
    if request.method == "GET":
        data = _read_json(path, default)
        return jsonify(data)
    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be JSON object"}), 400
    out = {"mode": data.get("mode", default["mode"]), "index": int(data.get("index", 0))}
    _write_json(path, out)
    return jsonify(out)


def _build_run_daily_cmd(user_id: str, output_dir: Optional[Path] = None) -> tuple[list[str], dict]:
    """为指定用户构建 run_daily 的 cmd 和 env。output_dir 默认 USERS_DIR / user_id。"""
    user_out_dir = output_dir or (USERS_DIR / user_id)
    report_date = _yesterday_yyyymmdd()

    settings_path = REPORTS / f"user_settings_{user_id}.json"
    settings = _read_json(settings_path, _default_user_settings())
    modules = _default_user_settings()["modules"]
    if isinstance(settings, dict) and isinstance(settings.get("modules"), dict):
        modules = {**modules, **settings["modules"]}
    tts_chunking = _default_user_settings()["tts_sync_chunking"]
    if isinstance(settings, dict) and settings.get("tts_sync_chunking") in ("atomic", "sentence"):
        tts_chunking = settings.get("tts_sync_chunking")

    enable = []
    if modules.get("digest"):
        enable.append("digest")
    if modules.get("podcast"):
        enable.append("podcast")
    if modules.get("sync_reader"):
        enable.append("tts_sync")
    if modules.get("japanese_points"):
        enable.append("japanese_points")

    interests_path = REPORTS / f"user_interests_{user_id}.json"
    themes = []
    if interests_path.exists():
        try:
            data = _read_json(interests_path, {})
            themes = data.get("themes") or []
            themes = [t for t in themes if isinstance(t, str) and t.strip()][:4]
        except Exception:
            pass

    cmd = [sys.executable, "-m", "run_daily"]
    if themes:
        cmd.extend(["--themes", ",".join(themes)])
    cmd.extend(["--enable", ",".join(enable)])
    cmd.extend(["--tts-sync-chunking", tts_chunking])
    log_file = user_out_dir / "logs" / f"{report_date}.log"
    cmd.extend(["--log-file", str(log_file)])

    env = dict(os.environ)
    env["OUTPUT_DIR"] = str(user_out_dir)
    return cmd, env


@app.route("/api/run_daily", methods=["POST"])
def api_run_daily():
    """登录用户触发每日流水线。按用户 settings 选择步骤，产物与日志写入该用户目录。后台启动子进程，立即返回。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401

    settings_path = REPORTS / f"user_settings_{user_id}.json"
    settings = _read_json(settings_path, _default_user_settings())
    modules = _default_user_settings()["modules"]
    if isinstance(settings, dict) and isinstance(settings.get("modules"), dict):
        modules = {**modules, **settings["modules"]}
    ok, msg = _validate_modules(modules)
    if not ok:
        return jsonify({"error": "invalid_modules", "message": msg}), 400

    cmd, env = _build_run_daily_cmd(user_id)
    report_date = _yesterday_yyyymmdd()
    interests_path = REPORTS / f"user_interests_{user_id}.json"
    themes = []
    if interests_path.exists():
        try:
            data = _read_json(interests_path, {})
            themes = data.get("themes") or []
            themes = [t for t in themes if isinstance(t, str) and t.strip()][:4]
        except Exception:
            pass
    try:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return jsonify({"error": "run_failed", "message": str(e)}), 500
    return jsonify({"ok": True, "message": "流水线已启动", "themes": themes, "report_date": report_date})


@app.route("/api/cron/run_daily", methods=["GET"])
def api_cron_run_daily():
    """供 Vercel Cron 在每天固定时间调用（如北京时间 1:00）。需设置 CRON_SECRET 与 CRON_USER_ID。"""
    auth = request.headers.get("Authorization")
    secret = os.environ.get("CRON_SECRET")
    if not secret or auth != f"Bearer {secret}":
        return jsonify({"error": "unauthorized"}), 401
    user_id = os.environ.get("CRON_USER_ID", "").strip()
    if not user_id:
        return jsonify({"error": "CRON_USER_ID not set"}), 500

    # Vercel serverless 仅 /tmp 可写，流水线输出写到 /tmp
    output_dir = Path(tempfile.gettempdir()) / "daily-news-digest" / "users" / user_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    cmd, env = _build_run_daily_cmd(user_id, output_dir=output_dir)
    timeout = int(os.environ.get("CRON_RUN_TIMEOUT", "300"))  # 默认 5 分钟，Vercel Pro 可设更大
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "timeout", "message": f"流水线超过 {timeout}s 未完成"}), 504
    except Exception as e:
        return jsonify({"ok": False, "error": "run_failed", "message": str(e)}), 500

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[-500:]
        return jsonify({"ok": False, "error": "pipeline_failed", "returncode": result.returncode, "stderr": err}), 500
    return jsonify({"ok": True, "message": "流水线已完成", "report_date": _yesterday_yyyymmdd()})


@app.route("/api/run_daily/log")
def api_run_daily_log():
    """返回当前登录用户某日流水线日志（用于前端轮询）。"""
    user_id = _current_user_id()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    date = request.args.get("date") or _yesterday_yyyymmdd()
    tail = request.args.get("tail") or ""
    try:
        tail_n = int(tail) if tail else 400
    except ValueError:
        tail_n = 400
    tail_n = max(50, min(2000, tail_n))
    log_path = USERS_DIR / user_id / "logs" / f"{date}.log"
    if not log_path.exists():
        return jsonify({"date": date, "exists": False, "content": ""})
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        content = "\n".join(lines[-tail_n:])
        return jsonify({"date": date, "exists": True, "content": content})
    except OSError as e:
        return jsonify({"error": "read_failed", "message": str(e)}), 500


if __name__ == "__main__":
    # use_reloader=False so OAuth state stored at /login is in the same process/session at /auth/callback
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)
