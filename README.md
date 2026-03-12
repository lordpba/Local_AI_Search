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

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Linux o Windows 10/11 | Linux (migliori performance GPU) |
| **Docker** | Docker Desktop installato e in esecuzione | |
| **Ollama** | Installato e in esecuzione ([ollama.com](https://ollama.com)) | |
| **GPU NVIDIA** | 1× GPU con 12 GB VRAM (profilo Veloce) | 2× GPU con 12 GB VRAM ciascuna |
| **RAM** | 16 GB | 32 GB |
| **Spazio disco** | ~25 GB per i modelli AI | |

### ⚠️ Note sulle GPU

| Configurazione GPU | Cosa funziona |
|---|---|
| **1× GPU 8 GB** (es. RTX 3060 8GB) | Solo profilo Veloce (gemma3:4b + gemma3:27b per OCR si alternano). Indicizzazione lenta. |
| **1× GPU 12 GB** (es. RTX 3060 12GB) | Profilo Veloce. OCR con gemma3:27b carica/scarica dalla VRAM. |
| **2× GPU 12 GB** (es. 2× RTX 3060) | **Configurazione ideale** — Profilo Massimo: `gemma3:27b` per tutto (OCR + Chat). Il modello si distribuisce su entrambe le GPU (~17 GB). |
| **1× GPU 24 GB** (es. RTX 3090/4090) | Tutti i profili incluso Massimo. Massime prestazioni con un solo modello. |

> **Profilo Massimo**: usa `gemma3:27b` (27B parametri, multimodale) sia per OCR che per Chat. Ollama deve caricare solo 2 modelli (bge-m3 + gemma3:27b) invece di 3, riducendo lo swap in VRAM e migliorando la velocità.

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

1. **⚙️ Setup** — Click "Check system", then "Download models" (~15 GB, first time only)
2. **📂 Documents** — Enter the path to your documents folder and click "Index"
3. **💬 Chat** — Start asking questions!

## 🎯 Profili

| Profilo | GPU richiesta | Modello Chat | Modello OCR | Caso d'uso |
|---|---|---|---|---|
| ⚡ Veloce | 6–8 GB VRAM | gemma3:4b | gemma3:27b | Risposte rapide, buona qualità |
| 🎯 Preciso | 12+ GB VRAM | gemma3:12b | gemma3:27b | Risposte più accurate e dettagliate |
| 🚀 Massimo | 20+ GB VRAM | gemma3:27b | gemma3:27b | **Un solo modello per tutto** — massima qualità |

### Modelli utilizzati

| Modello | Ruolo | Dimensione | VRAM richiesta |
|---|---|---|---|
| `bge-m3` | Embeddings (ricerca semantica) | ~2 GB | ~2 GB |
| `gemma3:27b` | OCR + Chat (profilo Massimo) | ~17 GB | ~17 GB (distribuibile su più GPU) |
| `gemma3:4b` | Chat — profilo Veloce | ~3 GB | ~4 GB |
| `gemma3:12b` | Chat — profilo Preciso | ~8 GB | ~10 GB |

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
│  │  (su host)   │   │  Container      │  │
│  │              │   │                 │  │
│  │ • gemma3:27b │◄──│ • Gradio UI     │  │
│  │  (OCR+Chat)  │   │ • ChromaDB      │  │
│  │ • bge-m3     │   │ • BM25 Index    │  │
│  │  (embeddings)│   │ • RAG Engine    │  │
│  │              │   │                 │  │
│  │  GPU ←───────│   │ (no GPU needed) │  │
│  └─────────────┘   └─────────────────┘  │
│                                         │
│  📂 I tuoi documenti (accesso sola lettura) │
└─────────────────────────────────────────┘
```

- **Ollama** gira sull'host e gestisce l'accesso GPU per OCR, embeddings e chat LLM
- **Il container Docker** è leggero (~500 MB), non richiede GPU ed è cross-platform
- **I tuoi documenti** sono montati in sola lettura — l'app non modifica mai i tuoi file
- **Ricerca ibrida**: semantica (ChromaDB + bge-m3) + keyword (BM25) con fusione RRF

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

*PrivateSearch v1.0 — I tuoi dati, il tuo dispositivo, la tua privacy.*
