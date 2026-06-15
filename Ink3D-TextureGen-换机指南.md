# Ink3D-TextureGen 换机开发指南

> 本文档记录了项目所有账号、配置和环境信息，帮助你在新电脑上无缝继续开发。

---

## 1. GitHub 账号信息

| 项目       | 值                                                                 |
| ---------- | ------------------------------------------------------------------ |
| **用户名** | `Ink3DTexGen`                                                      |
| **仓库名** | `Ink3D-TextureGen`                                                 |
| **仓库地址 (SSH)** | `git@github.com:Ink3DTexGen/Ink3D-TextureGen.git`           |
| **仓库地址 (HTTPS)** | `https://github.com/Ink3DTexGen/Ink3D-TextureGen.git`     |
| **GitHub Pages 网址** | https://ink3dtexgen.github.io/Ink3D-TextureGen/           |

---

## 2. 新电脑环境搭建步骤

### 2.1 安装必要工具

**Windows（推荐用 winget）：**

```powershell
winget install Git.Git
winget install GitHub.cli
```

**Mac：**

```bash
brew install git gh
```

**Linux：**

```bash
sudo apt install git gh
```

### 2.2 配置 Git 用户信息

```bash
git config --global user.name "Ink3DTexGen"
git config --global user.email "你的邮箱@example.com"
```

> ⚠️ 邮箱请填注册 GitHub 时用的邮箱，保持 commit 归属一致。

### 2.3 生成新的 SSH Key（每台电脑需要单独生成）

```bash
ssh-keygen -t ed25519 -C "你的邮箱@example.com"
```

一路回车即可（默认路径 `~/.ssh/id_ed25519`）。

然后复制公钥：

**Windows PowerShell：**
```powershell
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | Set-Clipboard
```

**Mac/Linux：**
```bash
cat ~/.ssh/id_ed25519.pub
# 手动复制输出内容
```

### 2.4 将公钥添加到 GitHub

1. 打开 https://github.com/settings/keys
2. 点击 **New SSH key**
3. Title 填写新电脑名称（如 `新笔记本`）
4. 粘贴刚才复制的公钥
5. 点击 **Add SSH key**

### 2.5 验证 SSH 连接

```bash
ssh -T git@github.com
```

看到 `Hi Ink3DTexGen! You've successfully authenticated` 即表示成功。

### 2.6 登录 GitHub CLI

```bash
gh auth login
```

选择：
- `GitHub.com`
- `SSH`
- 选择你的 SSH key
- `Login with a web browser`（按提示在浏览器中完成授权）

---

## 3. 克隆项目并开始开发

```bash
git clone git@github.com:Ink3DTexGen/Ink3D-TextureGen.git
cd Ink3D-TextureGen
```

---

## 4. 项目结构概览

```
Ink3D-TextureGen/
├── index.html              # 主页面
├── styles/
│   └── main.css            # 所有样式
├── scripts/
│   └── main.js             # 交互逻辑（灯箱、视频播放）
├── assets/
│   ├── videos/             # 展示视频 + 缩略图 (mp4, jpg)
│   └── images/             # Teaser 图片 (reallife_v4.jpg, Dream1.png)
├── fonts/                  # 自托管字体 (Baguet Script, Book Antiqua)
└── .github/
    └── workflows/
        └── pages.yml       # GitHub Pages 部署工作流
```

---

## 5. 部署方式

- **自动部署**：推送到 `main` 分支后，GitHub Actions 会自动触发部署。
- **工作流文件**：`.github/workflows/pages.yml`
- **无需任何构建工具**：纯静态站点（HTML + CSS + JS），无 Node.js/npm 依赖。

### 日常开发流程

```bash
# 编辑文件后
git add .
git commit -m "描述你的改动"
git push
# 等 1-2 分钟后刷新 Pages 网址即可看到更新
```

---

## 6. 技术栈备忘

| 类别     | 技术 / 工具                          |
| -------- | ------------------------------------ |
| 前端     | 纯 HTML + CSS + JS（无框架）         |
| 字体     | Baguet Script, Book Antiqua（自托管）|
| 布局     | CSS Grid, Flexbox                    |
| 部署     | GitHub Pages + Actions               |
| 版本控制 | Git + SSH                            |

---

## 7. 注意事项

1. **SSH Key 不要拷贝**：每台电脑应生成自己的密钥对，旧电脑的 key 可以在 GitHub Settings 中删除。
2. **大文件注意**：`Dream1.png` 约 5MB，如果后续图片/视频变多，考虑压缩或使用 Git LFS。
3. **字体文件已在仓库中**：`fonts/` 目录包含自托管字体，克隆后即可用，无需额外下载。
4. **浏览器预览**：开发时可以直接双击 `index.html` 在浏览器打开预览。

---

*文档生成日期：2026-06-15*
