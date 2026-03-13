"""
OCREngine — Extracts text from images using a vision model via Ollama API.
Handles batch processing with caching and image preprocessing for better handwriting recognition.
"""

import base64
import io
import json
import hashlib
import logging
from pathlib import Path

import httpx
from PIL import Image, ImageEnhance, ImageFilter

from app.config import OLLAMA_HOST, OCR_PROMPT, OCR_CACHE_DIR, get_active_model

logger = logging.getLogger(__name__)


def _preprocess_image(image_data: bytes) -> bytes:
    """
    Preprocess image for better OCR on handwritten text.
    Enhances contrast, sharpens, and converts to clean RGB.
    """
    try:
        img = Image.open(io.BytesIO(image_data))

        # Convert to grayscale for enhancement
        gray = img.convert("L")

        # Enhance contrast (helps faded handwriting ink)
        gray = ImageEnhance.Contrast(gray).enhance(1.6)

        # Sharpen (helps blurry scans)
        gray = gray.filter(ImageFilter.SHARPEN)

        # Back to RGB (vision models expect color input)
        img = gray.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image preprocessing failed, using original: {e}")
        return image_data


class OCREngine:
    """OCR via vision model on Ollama."""

    def __init__(self):
        self.cache_dir = OCR_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def api_url(self) -> str:
        from app.config import OLLAMA_HOST
        return f"{OLLAMA_HOST}/api/chat"

    def _cache_key(self, image_data: bytes) -> str:
        return hashlib.sha256(image_data).hexdigest()

    def _get_cached(self, cache_key: str) -> str | None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                return data.get("text")
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    def _save_cache(self, cache_key: str, text: str, metadata: dict):
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump({"text": text, "metadata": metadata}, f, ensure_ascii=False, indent=2)

    def ocr_image(self, image_data: bytes, metadata: dict | None = None) -> str:
        """
        Extract text from a single image using vision model via Ollama.

        Args:
            image_data: Raw image bytes (PNG/JPG)
            metadata: Optional metadata for cache context

        Returns:
            Extracted text string
        """
        cache_key = self._cache_key(image_data)

        # Check cache first
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"OCR cache hit: {cache_key[:12]}...")
            return cached

        # Preprocess image for better handwriting recognition
        processed = _preprocess_image(image_data)

        # Encode image to base64
        img_b64 = base64.b64encode(processed).decode("utf-8")

        # Call Ollama API
        payload = {
            "model": get_active_model(),
            "messages": [
                {
                    "role": "user",
                    "content": OCR_PROMPT,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096,
            },
        }

        try:
            response = httpx.post(
                self.api_url,
                json=payload,
                timeout=300.0,  # Vision models need more time on large images
            )
            response.raise_for_status()
            result = response.json()
            text = result.get("message", {}).get("content", "").strip()

            # Cache result
            self._save_cache(cache_key, text, metadata or {})
            logger.info(f"OCR completed: {len(text)} chars extracted")
            return text

        except httpx.TimeoutException:
            logger.error("OCR timeout — image may be too large or Ollama is overloaded")
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(f"OCR API error: {e.response.status_code} — {e.response.text}")
            return ""
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return ""

    def process_ocr_queue(
        self,
        ocr_queue: list[dict],
        progress_callback=None,
    ) -> list:
        """
        Process a batch of images through OCR.

        Args:
            ocr_queue: List of dicts with keys: image_data, filepath, filename, page_num
            progress_callback: Optional callback(progress_float, status_str)

        Returns:
            List of Document objects with extracted text
        """
        from app.indexer.document_loader import Document

        documents = []
        total = len(ocr_queue)

        if total == 0:
            return documents

        logger.info(f"Starting OCR on {total} images...")

        for i, item in enumerate(ocr_queue):
            if progress_callback:
                progress_callback(
                    i / total,
                    f"OCR: {item['filename']} (pag. {item['page_num']}) — {i + 1}/{total}"
                )

            text = self.ocr_image(
                image_data=item["image_data"],
                metadata={
                    "filepath": item["filepath"],
                    "filename": item["filename"],
                    "page": item["page_num"],
                },
            )

            if text and len(text.strip()) > 5:
                documents.append(Document(
                    text=text,
                    metadata={
                        "filepath": item["filepath"],
                        "filename": item["filename"],
                        "file_type": "ocr",
                        "page": item["page_num"],
                        "extraction": "ocr-vision",
                    }
                ))
            else:
                logger.warning(
                    f"OCR returned empty text for {item['filename']} page {item['page_num']}"
                )

        if progress_callback:
            progress_callback(1.0, f"OCR completato: {len(documents)}/{total} pagine estratte")

        return documents

    def is_available(self) -> bool:
        """Check if Ollama and OCR vision model are available."""
        try:
            from app.config import OLLAMA_HOST
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            # Check if our OCR model is present (handle tag variations)
            ocr_model = get_active_model()
            return any(m == ocr_model or m == f"{ocr_model}:latest" or (":" not in ocr_model and m.startswith(f"{ocr_model}:")) for m in models)
        except Exception:
            return False

    def clear_cache(self):
        """Clear all OCR cache files."""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            logger.info("OCR cache cleared")
