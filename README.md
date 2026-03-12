# 🔒 PrivateSearch

**100% local and private document search with AI.**

PrivateSearch lets you search and chat with your personal documents using artificial intelligence, without ever sending data over the network. Everything is processed locally on your machine.

## ✨ Features

- 🔍 **Hybrid search** — semantic search (ChromaDB + bge-m3) combined with keyword search (BM25), fused via Reciprocal Rank Fusion (RRF)
- 🤖 **Chat with your documents** — ask questions in natural language, get cited answers
- 📄 **Multi-format support** — PDF, DOCX, TXT, CSV, images (JPG, PNG, TIFF…)
- 👁️ **Intelligent OCR** — extracts text from images and scanned PDFs via multimodal LLM (handwriting-aware)
- 🧠 **Unified model family** — Qwen3.5 for everything: chat, OCR and reasoning, with 3 size profiles
- 🔄 **Incremental updates** — automatically detects new, modified, or deleted files
- 🖥️ **Cross-platform** — works on Linux and Windows (Docker + Ollama)
- 🛡️ **100% offline** — no internet connection required after initial setup

## 📋 Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Linux or Windows 10/11 | Linux (best GPU performance) |
| **Docker** | Docker Desktop installed and running | |
| **Ollama** | Installed and running ([ollama.com](https://ollama.com)) | |
| **NVIDIA GPU** | 1× GPU with 6 GB VRAM (Fast profile) | 2× GPUs with 12 GB VRAM each |
| **RAM** | 16 GB | 32 GB |
| **Disk space** | ~25 GB for AI models | |

## 🧠 Why Qwen3.5 (Unified Model)

PrivateSearch uses the **Qwen3.5** family as a unified model for all operations:

- **Chat & RAG** — answers questions based on indexed documents
- **Multimodal OCR** — reads images and scanned PDFs, including handwritten text
- **Reasoning** — understands complex queries and aggregates information from multiple sources

This "one model for everything" approach brings real advantages:

| Advantage | Detail |
|---|---|
| **Less GPU swapping** | Ollama loads only 2 models (bge-m3 + qwen3.5) instead of 3 separate ones |
| **Consistent OCR + Chat** | The same model that reads the document answers your questions |
| **Scalable profiles** | Same model in 3 sizes: 4B, 9B, 27B — choose based on your GPU |
| **Multilingual** | Qwen3.5 excels in English, Italian, and many other languages |

### ⚠️ GPU Notes

| GPU Configuration | Recommended profile |
|---|---|
| **1× GPU 6–8 GB** (e.g. RTX 3060 8GB) | ⚡ Fast (qwen3.5:4b) — quick answers, good quality |
| **1× GPU 12 GB** (e.g. RTX 3060 12GB) | 🎯 Precise (qwen3.5:9b) — accurate and detailed answers |
| **2× GPU 12 GB** (e.g. 2× RTX 3060) | 🚀 Maximum (qwen3.5:27b) — highest quality, model is distributed across both GPUs |
| **1× GPU 24 GB** (e.g. RTX 3090/4090) | 🚀 Maximum (qwen3.5:27b) — top performance with a single GPU |

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
2. Configure Docker so the container can reach Ollama on the host
3. Build and start the container
4. Open the browser at http://localhost:7860

If the browser does not open automatically, visit:

```text
http://localhost:7860
```

### 3. In-app setup

1. **⚙️ Setup** — Click "Check system", then "Download models" (~25 GB, first time only)
2. **📂 Documents** — Enter the path to your documents folder (Linux or Windows format) and click "Check", then "Index documents"
3. **💬 Chat** — Start asking questions!

## 🎯 Profiles

All profiles use **Qwen3.5** — same model, different sizes:

| Profile | Required GPU | Model (Chat + OCR) | Size | Use case |
|---|---|---|---|---|
| ⚡ Fast | 4–6 GB VRAM | `qwen3.5:4b` | ~3.4 GB | Quick answers, good quality |
| 🎯 Precise | 8–10 GB VRAM | `qwen3.5:9b` | ~6.6 GB | Accurate and detailed answers |
| 🚀 Maximum | 20+ GB VRAM | `qwen3.5:27b` | ~17 GB | **One model for everything** — highest quality |

> With the **Maximum** profile, the same `qwen3.5:27b` handles both OCR and Chat. Ollama loads only 2 models in VRAM (bge-m3 + qwen3.5:27b), eliminating swapping and greatly improving speed.

### Models used

| Model | Role | Size | Notes |
|---|---|---|---|
| `bge-m3` | Embeddings (semantic search) | ~2 GB | Always loaded, 1024 dimensions |
| `qwen3.5:4b` | Chat + OCR — Fast profile | ~3.4 GB | Multimodal (text + images) |
| `qwen3.5:9b` | Chat + OCR — Precise profile | ~6.6 GB | Multimodal (text + images) |
| `qwen3.5:27b` | Chat + OCR — Maximum profile | ~17 GB | Multimodal, can be distributed across multiple GPUs |

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
│  │ • qwen3.5    │◄──│ • Gradio UI     │  │
│  │  (Chat+OCR)  │   │ • ChromaDB      │  │
│  │ • bge-m3     │   │ • BM25 Index    │  │
│  │  (embeddings)│   │ • RAG Engine    │  │
│  │              │   │ • RRF Fusion    │  │
│  │  GPU ←───────│   │ (no GPU needed) │  │
│  └─────────────┘   └─────────────────┘  │
│                                         │
│  📂 Your documents (read-only access)    │
└─────────────────────────────────────────┘
```

### Search pipeline

```
User query
    │
    ├─► Semantic Search (ChromaDB + bge-m3)  ─┐
    │                                          ├─► RRF Fusion ─► Top-K chunks ─► LLM (qwen3.5)
    └─► Keyword Search (BM25)                ─┘
```

- **Ollama** runs on the host and manages GPU access for OCR, embeddings, and chat LLM
- **The Docker container** is lightweight (~500 MB), requires no GPU, and is cross-platform (Linux/Windows)
- **Your documents** are mounted read-only — the app never modifies your files
- **Hybrid search**: semantic (ChromaDB + bge-m3) + keyword (BM25) with RRF fusion (k=60)
- **Map-Reduce**: for aggregation queries ("list all…"), iterates all files in batches of 5

### Smart OCR

OCR uses the same Qwen3.5 model as the active profile, with:

- **Image pre-processing**: grayscale conversion → contrast (1.6×) → sharpening
- **PDF at 300 DPI**: high-res rendering to capture handwritten text
- **Italian OCR prompt**: optimized for government forms, tax codes, dates, IBANs
- **Persistent cache**: each page is processed only once and saved to disk

## 🌐 Cross-platform: Linux + Windows

PrivateSearch works on both operating systems:

- **Document paths**: you can enter either `/home/mario/Documents` (Linux) or `C:\Users\mario\Documents` (Windows) — normalization is automatic
- **Docker networking**: uses bridge network + `host.docker.internal` to reach Ollama on the host, compatible with Docker Desktop (Windows) and Docker Engine (Linux)
- **Remote Ollama**: you can configure a remote Ollama server (e.g. `http://192.168.1.100:11434`) from the Config tab

> **Windows note**: Ollama must listen on `0.0.0.0` (not just `127.0.0.1`) to be reachable from the Docker container.

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
```
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

*PrivateSearch v1.0 — I tuoi dati, il tuo dispositivo, la tua privacy.*
