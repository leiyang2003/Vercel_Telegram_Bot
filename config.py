"""配置：从环境变量读取。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")

# 项目根目录，用户数据目录
ROOT = Path(__file__).resolve().parent
USERS_DIR = ROOT / "users"
