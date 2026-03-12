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
# In Docker we usually point this to host.docker.internal so the container can
# reach the Ollama instance running on the host. Outside Docker, localhost is
# still the sensible default.

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# ─── Model definitions ───────────────────────────────────────────────────────

OCR_MODEL = "deepseek-ocr:latest"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024  # bge-m3 output dimension

# Chat model profiles
PROFILES = {
    "fast": {
        "name": "⚡ Veloce",
        "description": "GPU 6-8 GB · Risposte rapide",
        "model": "gemma3:4b",
        "gpu_min_gb": 6,
    },
    "precise": {
        "name": "🎯 Preciso",
        "description": "GPU 12 GB+ · Risposte più accurate",
        "model": "gemma3:12b",
        "gpu_min_gb": 12,
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
    "Extract ALL visible text from this image. "
    "Preserve the original structure including headers, paragraphs, lists, and tables. "
    "Output in clean markdown format. "
    "IMPORTANT: Preserve ALL alphanumeric codes EXACTLY as they appear, character by character. "
    "This includes: codici fiscali (e.g. RSSMRA80A01H501Z), IBAN, dates, protocol numbers, "
    "document identifiers, and any other codes or identifiers. "
    "Maintain key-value pairs from form fields (e.g. 'Codice Fiscale: RSSMRA80A01H501Z'). "
    "If handwritten text is present, transcribe it as accurately as possible. "
    "Do not add any commentary, only output the extracted text."
)

# Minimum text length from PDF native extraction to skip OCR
PDF_MIN_TEXT_LENGTH = 50

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
    hp = host_path.replace("\\", "/").rstrip("/")
    home = HOST_HOME_PATH.replace("\\", "/").rstrip("/")
    if hp.startswith(home):
        return CONTAINER_HOME_MOUNT + hp[len(home):]
    return host_path  # Path outside HOME — can't translate


def container_to_host_path(container_path: str) -> str:
    """Translate a container-mapped path back to the host path (for display)."""
    if not HOST_HOME_PATH:
        return container_path
    cp = container_path.replace("\\", "/").rstrip("/")
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
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError):
                pass
        return cls()

    @property
    def chat_model(self) -> str:
        return PROFILES.get(self.profile, PROFILES["fast"])["model"]

    @property
    def required_models(self) -> list[str]:
        return [OCR_MODEL, EMBEDDING_MODEL, self.chat_model]
