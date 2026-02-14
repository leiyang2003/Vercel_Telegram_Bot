"""配置：从环境变量读取 API Key、超时、代理。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env
load_dotenv(Path(__file__).resolve().parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)))

# 请求超时（秒）。agentic web_search 可能需 10–30 分钟，可设大一些或通过 DIGEST_TIMEOUT 覆盖
DIGEST_TIMEOUT = int(os.getenv("DIGEST_TIMEOUT", "3600"))  # 默认 1 小时

# 「当日」采用的时区（IANA，如 Asia/Shanghai），避免中美时差歧义
DIGEST_TIMEZONE = os.getenv("DIGEST_TIMEZONE", "Asia/Shanghai")

# 可选代理，例如 http://127.0.0.1:7890 或 socks5://...
HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
PROXIES = None
if HTTP_PROXY or HTTPS_PROXY:
    PROXIES = {"http": HTTP_PROXY or HTTPS_PROXY, "https": HTTPS_PROXY or HTTP_PROXY}

# TTS：OpenAI 语音合成（播客音频）
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1-hd")
TTS_VOICE = os.getenv("TTS_VOICE", "nova")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# Google OAuth（温故知新等需登录功能）
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:5003/auth/callback")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
