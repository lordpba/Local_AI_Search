"""
PrivateSearch — Configuration
All settings for the application. Profiles, models, chunking, paths.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Base paths ───────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = Path(os.environ.get("PRIVATESEARCH_DATA", str(PROJECT_DIR / "data")))
CHROMA_DIR = DATA_DIR / "chromadb"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
CONFIG_FILE = DATA_DIR / "config.json"
MANIFEST_FILE = DATA_DIR / "index_manifest.json"

# ─── Ollama connection ────────────────────────────────────────────────────────
# With network_mode: host, the container shares the host's network stack,
# so Ollama is reachable at localhost:11434 (its default bind address).

_DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Mutable runtime value — updated by UserConfig.load()
OLLAMA_HOST = _DEFAULT_OLLAMA_HOST


def get_ollama_host() -> str:
    """Return the current Ollama host URL."""
    return OLLAMA_HOST


def set_ollama_host(url: str):
    """Update the Ollama host URL at runtime."""
    global OLLAMA_HOST
    OLLAMA_HOST = url.rstrip("/")

# ─── Model definitions ───────────────────────────────────────────────────────

# Unified model family: qwen3.5 handles BOTH chat and OCR (multimodal)
# The actual model size depends on the selected profile
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 output dimension

# Legacy constant — use get_active_model() for profile-aware access
OCR_MODEL = "qwen3.5:27b"

# Chat model profiles — all qwen3.5 (unified vision + language)
PROFILES = {
    "fast": {
        "name": "⚡ Veloce",
        "description": "GPU 4 GB · Risposte rapide",
        "model": "qwen3.5:4b",
        "gpu_min_gb": 4,
    },
    "precise": {
        "name": "🎯 Preciso",
        "description": "GPU 8 GB · Risposte accurate",
        "model": "qwen3.5:9b",
        "gpu_min_gb": 8,
    },
    "maximum": {
        "name": "🚀 Massimo",
        "description": "2× GPU 12 GB · Un solo modello per tutto (OCR + Chat)",
        "model": "qwen3.5:27b",
        "gpu_min_gb": 20,
    },
}

# ─── Chunking settings ────────────────────────────────────────────────────────

CHUNK_SIZE = 768          # tokens (chars approximation: *4)
CHUNK_OVERLAP = 128       # tokens overlap between chunks
CHUNK_SIZE_CHARS = CHUNK_SIZE * 4       # ~3072 chars
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP * 4  # ~512 chars

# ─── Retrieval settings ───────────────────────────────────────────────────────

TOP_K = 12                # Chunks to retrieve for normal queries
TOP_K_AGGREGATION = 50    # Chunks to retrieve for aggregation queries
SIMILARITY_THRESHOLD = 0.25  # Minimum similarity score (0-1)
RRF_K = 60                # Reciprocal Rank Fusion constant
BM25_INDEX_FILE = DATA_DIR / "bm25_index.pkl"

# ─── LLM settings ─────────────────────────────────────────────────────────────

TEMPERATURE = 0.1         # Low temperature = less hallucination
MAX_TOKENS = 4096         # Max response tokens (normal)
MAX_TOKENS_AGGREGATION = 8192  # Max response tokens (aggregation)

# ─── Supported file types ─────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {
    "text": [".txt", ".md", ".csv", ".json", ".xml", ".html", ".log"],
    "document": [".pdf", ".docx", ".doc", ".odt", ".rtf"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"],
}

ALL_EXTENSIONS = set()
for exts in SUPPORTED_EXTENSIONS.values():
    ALL_EXTENSIONS.update(exts)

# ─── OCR settings ─────────────────────────────────────────────────────────────

OCR_PROMPT = (
    "Sei un sistema OCR specializzato per moduli e documenti ufficiali italiani.\n"
    "Questa immagine contiene un modulo/documento che può avere sia testo stampato che TESTO SCRITTO A MANO.\n\n"
    "ISTRUZIONI:\n"
    "1. Trascrivi TUTTO il testo visibile, sia stampato che manoscritto.\n"
    "2. Per il testo manoscritto: leggi con estrema attenzione ogni singola lettera e cifra. "
    "I nomi propri (nome, cognome) e i codici fiscali sono PARTICOLARMENTE IMPORTANTI.\n"
    "3. Mantieni la struttura del documento con i campi nel formato 'CAMPO: valore'.\n"
    "4. I codici fiscali italiani hanno 16 caratteri alfanumerici (es. RSSMRA80A01H501Z): "
    "trascrivili ESATTAMENTE.\n"
    "5. Trascrivi numeri di telefono, email, PEC, IBAN, partite IVA, date e protocolli "
    "esattamente come appaiono.\n"
    "6. Se una parola manoscritta è incerta, scrivi la tua migliore lettura.\n"
    "7. NON aggiungere commenti, spiegazioni o note. Solo il testo estratto.\n"
    "8. NON usare markdown. Usa testo semplice."
)

# Minimum text length from PDF native extraction to skip OCR
PDF_MIN_TEXT_LENGTH = 50

# PDF render DPI for OCR (300 = standard for handwritten text)
PDF_OCR_DPI = 300

# ─── Gradio settings ──────────────────────────────────────────────────────────

GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860
APP_TITLE = "PrivateSearch"
APP_SUBTITLE = "🔒 Ricerca documenti 100% locale e privata"

# ─── Path mapping (Docker ↔ Host) ─────────────────────────────────────────────

# When running in Docker, the host's HOME is mounted at /host-home.
# HOST_HOME_PATH tells us what $HOME was on the host so we can translate paths.
HOST_HOME_PATH = os.environ.get("HOST_HOME_PATH", "")
CONTAINER_HOME_MOUNT = "/host-home"


def host_to_container_path(host_path: str) -> str:
    """Translate a host filesystem path to the container-mapped path."""
    if not HOST_HOME_PATH:
        return host_path  # Not in Docker or no mapping
    hp = host_path.rstrip("/")
    home = HOST_HOME_PATH.rstrip("/")
    if hp.startswith(home):
        return CONTAINER_HOME_MOUNT + hp[len(home):]
    return host_path  # Path outside HOME — can't translate


def container_to_host_path(container_path: str) -> str:
    """Translate a container-mapped path back to the host path (for display)."""
    if not HOST_HOME_PATH:
        return container_path
    cp = container_path.rstrip("/")
    mount = CONTAINER_HOME_MOUNT.rstrip("/")
    if cp.startswith(mount):
        return HOST_HOME_PATH.rstrip("/") + cp[len(mount):]
    return container_path


# ─── User config persistence ─────────────────────────────────────────────────

@dataclass
class UserConfig:
    """Persisted user configuration."""
    profile: str = "fast"
    folder_path: str = ""
    ollama_host: str = ""
    first_run_done: bool = False
    models_downloaded: bool = False

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "UserConfig":
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                cfg = cls()
        else:
            cfg = cls()
        # Apply saved Ollama host if present
        if cfg.ollama_host:
            set_ollama_host(cfg.ollama_host)
        return cfg

    @property
    def chat_model(self) -> str:
        return PROFILES.get(self.profile, PROFILES["fast"])["model"]

    @property
    def required_models(self) -> list[str]:
        """Embedding + the profile's unified model (chat & OCR are the same)."""
        models = {EMBEDDING_MODEL, self.chat_model}
        return sorted(models)


def get_active_model() -> str:
    """Get the active qwen3.5 model for the current profile.
    Since qwen3.5 is unified (vision + language), this is used for BOTH chat and OCR."""
    config = UserConfig.load()
    return config.chat_model
