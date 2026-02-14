"""
多用户 Telegram Bot Setup — Flask + Google OAuth。
启动后：未登录 → 登录页；登录后 → Setup UI。
"""
import json
import os
import re
from pathlib import Path

from authlib.integrations.flask_client import OAuth
from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session

from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, ROOT, SECRET_KEY
from bot_service import (
    SLOT_IDS,
    get_agents_for_display,
    get_agent_detail,
    get_agent_by_webhook_secret,
    save_agent,
    export_run_command,
    get_run_package,
    register_webhook,
)

app = Flask(__name__, static_folder=None)
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# HTTPS 时使用 Secure cookie（Vercel 默认 HTTPS）
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("VERCEL", "") == "1"

# Google OAuth
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
    if not sub:
        return ""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(sub)).strip() or "unknown"


def _current_user_id():
    return session.get("user_id")


def _require_login():
    if not _current_user_id():
        return None
    return _current_user_id()


# --- Routes ---


@app.route("/")
def index():
    if not _current_user_id():
        return redirect("/login?next=/setup")
    return redirect("/setup")


@app.route("/login")
def login():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return redirect("/login.html?error=oauth_not_configured")
    session["next_after_login"] = request.args.get("next", "/setup")
    redirect_uri = request.url_root.rstrip("/") + "/auth/callback"
    return oauth.google.authorize_redirect(redirect_uri)


@app.route("/auth/callback")
def auth_callback():
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
    next_url = session.pop("next_after_login", "/setup")
    return redirect(next_url)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/api/me")
def api_me():
    uid = _current_user_id()
    if not uid:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "email": session.get("email") or "",
        "name": session.get("name") or "",
    })


@app.route("/login.html")
def login_page():
    """登录页（含 OAuth 未配置时的提示）"""
    return send_from_directory(ROOT, "login.html")


@app.route("/setup")
@app.route("/setup.html")
def setup_page():
    if not _current_user_id():
        return redirect("/login?next=/setup")
    return send_from_directory(ROOT, "setup.html")


# --- Bot API ---


@app.route("/api/bots", methods=["GET"])
def api_bots_list():
    user_id = _require_login()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    agents = get_agents_for_display(user_id)
    return jsonify({"agents": agents, "slot_ids": SLOT_IDS})


@app.route("/api/bots/<slot_id>", methods=["GET"])
def api_bots_get(slot_id):
    user_id = _require_login()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    if slot_id not in SLOT_IDS:
        return jsonify({"error": "invalid_slot"}), 400
    agent = get_agent_detail(user_id, slot_id)
    if not agent:
        return jsonify({"id": slot_id, "name": "", "telegram_bot_token": "", "persona_text": "You are a helpful assistant.", "persona_filename": "persona.txt", "workspace": ""})
    return jsonify(agent)


@app.route("/api/bots/<slot_id>", methods=["POST"])
def api_bots_save(slot_id):
    user_id = _require_login()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    if slot_id not in SLOT_IDS:
        return jsonify({"error": "invalid_slot"}), 400
    data = request.get_json(force=True, silent=True) or {}
    try:
        ok, msg = save_agent(
            user_id,
            slot_id,
            data.get("name", ""),
            data.get("telegram_bot_token", ""),
            data.get("persona_text", ""),
            data.get("persona_filename", "persona.txt"),
        )
    except Exception as e:
        return jsonify({"error": f"Storage error: {str(e)}"}), 500
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "message": msg})


@app.route("/api/bots/<slot_id>/export", methods=["GET"])
def api_bots_export(slot_id):
    """导出运行命令（用于本地运行）"""
    user_id = _require_login()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    if slot_id not in SLOT_IDS:
        return jsonify({"error": "invalid_slot"}), 400
    ok, msg = export_run_command(user_id, slot_id)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "command": msg})


@app.route("/api/bots/<slot_id>/download", methods=["GET"])
def api_bots_download(slot_id):
    """下载运行配置包（JSON，含 persona 与 env 模板，便于 Blob 用户本地部署）"""
    user_id = _require_login()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    if slot_id not in SLOT_IDS:
        return jsonify({"error": "invalid_slot"}), 400
    pkg = get_run_package(user_id, slot_id)
    if not pkg:
        return jsonify({"error": "Agent not found"}), 404
    body = json.dumps(pkg, ensure_ascii=False, indent=2)
    return Response(
        body,
        mimetype="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=bot_config_{slot_id}.json",
        },
    )


@app.route("/api/bots/<slot_id>/register-webhook", methods=["POST"])
def api_bots_register_webhook(slot_id):
    """Register Telegram webhook for this bot (requires login)."""
    user_id = _require_login()
    if not user_id:
        return jsonify({"error": "login_required"}), 401
    if slot_id not in SLOT_IDS:
        return jsonify({"error": "invalid_slot"}), 400
    base_url = (request.get_json(silent=True) or {}).get("base_url", "").strip()
    if not base_url:
        base_url = os.environ.get("VERCEL_URL", "").strip()
        if base_url and not base_url.startswith("http"):
            base_url = f"https://{base_url}"
    if not base_url:
        return jsonify({"error": "base_url or VERCEL_URL required"}), 400
    ok, msg = register_webhook(user_id, slot_id, base_url)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True, "message": msg})


@app.route("/api/telegram/webhook/<webhook_secret>", methods=["POST"])
def api_telegram_webhook(webhook_secret):
    """Receive Telegram updates (no auth; path secret is the credential)."""
    lookup = get_agent_by_webhook_secret(webhook_secret)
    if not lookup:
        return "", 404
    user_id, slot_id, agent = lookup
    try:
        data = request.get_json(force=True, silent=True) or request.get_data(as_text=True)
        if isinstance(data, str) and data.strip():
            import json as _json
            data = _json.loads(data)
        if not data:
            return "", 200
    except Exception:
        return "", 200
    try:
        from webhook_handler import handle_webhook_update
        handle_webhook_update(user_id, slot_id, agent, data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            from webhook_handler import _send_error_fallback
            _send_error_fallback(agent.get("telegram_bot_token"), data, str(e))
        except Exception:
            pass
    return "", 200


# --- Static (Vercel 自动从 public/ 提供 /base.css 等；本地由 Flask 提供) ---


@app.route("/base.css")
@app.route("/auth.js")
def serve_static_root():
    """本地开发时提供 public/ 下根路径静态文件"""
    name = request.path.lstrip("/")
    return send_from_directory(ROOT / "public", name)


@app.route("/public/<path:path>")
def serve_public(path):
    return send_from_directory(ROOT / "public", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)
