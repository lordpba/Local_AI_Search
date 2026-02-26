# 🔒 PrivateSearch

**100% local and private document search with AI.**

PrivateSearch lets you search and chat with your personal documents using artificial intelligence, without ever sending data over the network. Everything is processed locally on your machine.

## ✨ Features

- 🔍 **Semantic search** — find documents by meaning, not just keywords
- 🤖 **Chat with your documents** — ask questions in natural language
- 📄 **Multi-format support** — PDF, DOCX, TXT, CSV, images (JPG, PNG, TIFF…)
- 👁️ **Intelligent OCR** — extracts text from images and scanned documents via multimodal LLM
- 🔄 **Incremental updates** — automatically detects new, modified, or deleted files
- 🛡️ **100% offline** — no internet connection required after initial setup

## 📋 Requirements

| Component | Requirement |
|---|---|
| **OS** | Linux or Windows 10/11 |
| **Docker** | Docker Desktop installed and running |
| **Ollama** | Installed and running ([ollama.com](https://ollama.com)) |
| **GPU** | NVIDIA/AMD with at least 6 GB VRAM (Fast profile) or 12 GB (Precise profile) |
| **RAM** | 8 GB minimum |
| **Disk space** | ~15 GB for AI models |

## 🚀 Quick Start

### 1. Install prerequisites

```bash
# Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Launch PrivateSearch

**Linux:**
```bash
./start.sh
```

**Windows:**
```
Double-click start.bat
```

The launcher script will:
1. Check that Docker and Ollama are running
2. Configure Ollama for Docker access (requires `sudo` once on Linux)
3. Build and start the container
4. Open the browser at http://localhost:7860

### 3. In-app setup

1. **⚙️ Setup** — Click "Check system", then "Download models" (~15 GB, first time only)
2. **📂 Documents** — Enter the path to your documents folder and click "Index"
3. **💬 Chat** — Start asking questions!

## 🎯 Profiles

| Profile | GPU | Model | Use case |
|---|---|---|---|
| ⚡ Fast | 6–8 GB VRAM | gemma3:4b | Quick answers, good quality |
| 🎯 Precise | 12+ GB VRAM | gemma3:12b | More accurate and detailed answers |

## 📁 Supported Formats

| Category | Extensions |
|---|---|
| Text | `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.log` |
| Documents | `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf` |
| Images (OCR) | `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.webp` |

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Your machine                  │
│                                         │
│  ┌─────────────┐   ┌─────────────────┐  │
│  │   Ollama     │   │   Docker        │  │
│  │  (on host)   │   │  Container      │  │
│  │              │   │                 │  │
│  │ • deepseek-  │◄──│ • Gradio UI     │  │
│  │   ocr        │   │ • ChromaDB      │  │
│  │ • bge-m3     │   │ • Indexer       │  │
│  │ • gemma3     │   │ • RAG Engine    │  │
│  │              │   │                 │  │
│  │  GPU ←───────│   │ (no GPU needed) │  │
│  └─────────────┘   └─────────────────┘  │
│                                         │
│  📂 Your documents (read-only access)   │
└─────────────────────────────────────────┘
```

- **Ollama** runs on the host and handles GPU access for OCR, embeddings, and chat LLM
- **The Docker container** is lightweight (~500 MB), requires no GPU, and is cross-platform
- **Your documents** are mounted read-only — the app never modifies your files

## 🛑 Stop

```bash
# Linux
docker compose -f docker/docker-compose.yml down

# Windows
docker compose -f docker\docker-compose.yml down
```

## 🔧 Useful Commands

```bash
# View app logs
docker logs privatesearch -f

# Rebuild after code changes
docker compose -f docker/docker-compose.yml up -d --build

# Delete all data (index, cache, config)
docker compose -f docker/docker-compose.yml down -v
```

## 🔒 Privacy

- **Zero telemetry** — no data collected
- **Zero outbound connections** — everything processed locally
- **Zero cloud** — no external services
- Your documents are mounted read-only inside the container
- Source code is fully auditable and transparent

---

*PrivateSearch v1.0 — Your data, your device, your privacy.*
