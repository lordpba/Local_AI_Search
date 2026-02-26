# 🔒 PrivateSearch

**Ricerca documenti 100% locale e privata con AI.**

PrivateSearch permette di cercare e interrogare i tuoi documenti personali usando l'intelligenza artificiale, senza mai inviare dati in rete. Tutto viene elaborato localmente sul tuo computer.

## ✨ Funzionalità

- 🔍 **Ricerca semantica** — trova documenti per significato, non solo per parole chiave
- 🤖 **Chat con i tuoi documenti** — fai domande in linguaggio naturale
- 📄 **Supporto multi-formato** — PDF, DOCX, TXT, CSV, immagini (JPG, PNG, TIFF...)
- 👁️ **OCR intelligente** — estrae testo da immagini e documenti scansionati
- 🔄 **Aggiornamento incrementale** — rileva automaticamente file nuovi, modificati o eliminati
- 🛡️ **100% offline** — nessuna connessione internet richiesta dopo la configurazione iniziale

## 📋 Requisiti

| Componente | Requisito |
|---|---|
| **Sistema operativo** | Linux o Windows 10/11 |
| **Docker** | Docker Desktop installato e in esecuzione |
| **Ollama** | Installato e in esecuzione ([ollama.com](https://ollama.com)) |
| **GPU** | NVIDIA con almeno 6 GB VRAM (profilo Veloce) o 12 GB (profilo Preciso) |
| **RAM** | Almeno 8 GB |
| **Spazio disco** | ~15 GB per i modelli AI |

## 🚀 Installazione rapida

### 1. Installa i prerequisiti

```bash
# Docker Desktop
# Scarica da: https://www.docker.com/products/docker-desktop/

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Avvia PrivateSearch

**Linux:**
```bash
./start.sh
```

**Windows:**
```
Doppio clic su start.bat
```

Lo script:
1. Verifica Docker e Ollama
2. Configura Ollama per l'accesso Docker (richiede `sudo` una sola volta su Linux)
3. Costruisce e avvia il container
4. Apre il browser su http://localhost:7860

### 3. Configurazione nell'app

1. **⚙️ Configurazione** — Clicca "Verifica sistema", poi "Scarica modelli" (~15 GB, solo la prima volta)
2. **📂 Documenti** — Inserisci il percorso della cartella documenti e clicca "Indicizza"
3. **💬 Chat** — Inizia a fare domande!

## 🎯 Profili

| Profilo | GPU | Modello | Uso |
|---|---|---|---|
| ⚡ Veloce | 6-8 GB VRAM | gemma3:4b | Risposte rapide, buona qualità |
| 🎯 Preciso | 12+ GB VRAM | gemma3:12b | Risposte più accurate e dettagliate |

## 📁 Formati supportati

| Categoria | Estensioni |
|---|---|
| Testo | `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.log` |
| Documenti | `.pdf`, `.docx`, `.doc`, `.odt`, `.rtf` |
| Immagini (OCR) | `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.webp` |

## 🏗️ Architettura

```
┌─────────────────────────────────────────┐
│           Il tuo computer               │
│                                         │
│  ┌─────────────┐   ┌─────────────────┐  │
│  │   Ollama     │   │   Docker        │  │
│  │  (su host)   │   │  Container      │  │
│  │             │   │                 │  │
│  │ • deepseek- │◄──│ • Gradio UI     │  │
│  │   ocr       │   │ • ChromaDB      │  │
│  │ • bge-m3    │   │ • Indexer       │  │
│  │ • gemma3    │   │ • RAG Engine    │  │
│  │             │   │                 │  │
│  │  GPU ←──────│   │ (no GPU needed) │  │
│  └─────────────┘   └─────────────────┘  │
│                                         │
│  📂 I tuoi documenti (accesso sola lettura)  │
└─────────────────────────────────────────┘
```

- **Ollama** gira sull'host e gestisce l'accesso GPU per OCR, embeddings e chat LLM
- **Il container Docker** è leggero (~500 MB), non richiede GPU, ed è cross-platform
- **I documenti** vengono montati in sola lettura — l'app non modifica mai i tuoi file

## 🛑 Stop

```bash
# Linux
docker compose -f docker/docker-compose.yml down

# Windows
docker compose -f docker\docker-compose.yml down
```

## 🔧 Comandi utili

```bash
# Vedere i log dell'app
docker logs privatesearch -f

# Ricostruire dopo modifiche al codice
docker compose -f docker/docker-compose.yml up -d --build

# Eliminare tutti i dati (indice, cache, configurazione)
docker compose -f docker/docker-compose.yml down -v
```

## 🔒 Privacy

- **Zero telemetria** — nessun dato raccolto
- **Zero connessioni in uscita** — tutto elaborato localmente
- **Zero cloud** — nessun servizio esterno
- I tuoi documenti sono montati in sola lettura nel container
- Il codice sorgente è verificabile e trasparente

---

*PrivateSearch v1.0 — I tuoi dati, il tuo dispositivo, la tua privacy.*
