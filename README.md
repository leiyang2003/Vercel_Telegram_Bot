# Vercel Telegram Bot

多用户 Telegram Bot 系统，可部署于 Vercel。

- **独立项目**：与 `telegramBOT` 分开，不直接修改原项目
- **参考备份**：`_reference/` 内含 telegramBOT、DailyNewsDigest 相关文件备份，供开发参考
- **下载配置包**：部署于 Vercel（Blob）时，可点击「下载」获取 JSON 配置，便于本地运行
- **功能**：Google 登录 → Setup UI → 每位用户最多 4 个 Bot，Persona/Skill 按用户隔离
- **部署**：支持 Vercel（Serverless）+ Vercel Blob 存储

## 开发计划

详见 `.cursor/plans/` 中的多用户 Telegram Bot 系统改造计划。核心逻辑参考 `../telegramBOT`（QX.py、ChatBot_OpenClaw、setup_ui 等），在此项目中重新实现为多用户 Web 应用。

## 本地运行

```bash
cd Vercel_Telegram_Bot
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填写 GOOGLE_CLIENT_ID、GOOGLE_CLIENT_SECRET、SECRET_KEY
python app.py
```

浏览器打开 http://127.0.0.1:5003/ ，未登录会跳转 Google 登录，登录后进入 Setup 页管理 Bot。

### Google OAuth 配置

1. 在 [Google Cloud Console](https://console.cloud.google.com/) 创建 OAuth 2.0 客户端 ID（Web 应用）
2. 授权重定向 URI 添加：`http://127.0.0.1:5003/auth/callback`、`http://localhost:5003/auth/callback`
3. 将 Client ID、Client Secret 填入 .env

## Vercel 部署

### 1. 部署

```bash
vercel
```

或关联 GitHub 仓库在 [vercel.com](https://vercel.com) 自动部署。

### 2. 环境变量（Vercel Dashboard → Project → Settings → Environment Variables）

| 变量 | 说明 |
|------|------|
| `GOOGLE_CLIENT_ID` | Google OAuth 客户端 ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth 密钥 |
| `SECRET_KEY` | Flask 会话签名（随机长字符串） |
| `BLOB_READ_WRITE_TOKEN` | Vercel Blob 读写令牌（必填，否则数据不持久） |

### 3. Vercel Blob

1. 在 Vercel 项目中打开 **Storage** → **Create Database** → 选择 **Blob**
2. 创建后复制 `BLOB_READ_WRITE_TOKEN` 到环境变量
3. 用户配置、Persona 将持久化到 Blob

### 4. Google OAuth 回调

在 [Google Cloud Console](https://console.cloud.google.com/) → 凭据 → OAuth 2.0 客户端 → 已授权的重定向 URI 中添加：

- `https://你的项目.vercel.app/auth/callback`
- `https://你的自定义域名/auth/callback`（若使用自定义域名）
# Vercel_Telegram_Bot
