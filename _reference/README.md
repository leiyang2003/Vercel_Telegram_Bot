# 参考文件备份

此目录存放从原项目备份的文件，供 Vercel_Telegram_Bot 开发时参考。**请勿直接修改这些文件**，实现时在项目根目录创建新文件。

## telegramBOT/

| 文件 | 用途 |
|------|------|
| setup_ui.py | Gradio Setup UI 逻辑 → 抽取为 bot_service.py + Flask API |
| QX.py | Telegram bot 主入口 → 本地运行时用；Webhook 模式需拆出消息处理逻辑 |
| ChatBot_OpenClaw.py | 对话、Memory、Skill 逻辑 → Webhook 模式需在 serverless 中调用 |
| dynamic-prompt.py | Visual Snapshot、conversations 等 |
| telegram_bots_config.json | 配置结构参考 |
| personas/ | Persona 文件示例 |
| workspace/ | Workspace 结构（memory、skills）示例 |
| .env.example | 环境变量参考 |
| requirements.txt | 依赖参考 |

## DailyNewsDigest/

| 文件 | 用途 |
|------|------|
| app.py | Flask + Google OAuth、session、API 结构参考 |
| config.py | 配置读取方式 |
| index.html | 首页、导航结构 |
| settings.html | 需登录的设置页、API 调用示例 |
| static/auth.js | 登录状态展示、Login/Logout |
| static/base.css | 样式 token、通用类 |
