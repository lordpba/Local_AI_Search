"""
ChatEngine — RAG-based chat using Ollama LLM.
Combines retrieval + prompt construction + streaming response.
"""

import logging
from typing import Generator

import httpx

from app.config import OLLAMA_HOST, TEMPERATURE, MAX_TOKENS, UserConfig
from app.search.retriever import Retriever
from app.llm.prompts import build_messages

logger = logging.getLogger(__name__)


class ChatEngine:
    """RAG chat engine: retrieve context → build prompt → stream LLM response."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.api_url = f"{OLLAMA_HOST}/api/chat"
        self.chat_history: list[dict] = []

    @property
    def model(self) -> str:
        return UserConfig.load().chat_model

    def ask(
        self,
        question: str,
        top_k: int = 8,
    ) -> Generator[tuple[str, list[dict]], None, None]:
        """
        Ask a question with RAG. Yields partial responses for streaming.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve

        Yields:
            (partial_response_text, sources) tuples
        """
        # 1. Retrieve relevant chunks
        results = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.format_context(results)
        sources = self.retriever.format_sources(results)

        # 2. Build prompt messages
        messages = build_messages(
            user_question=question,
            context=context,
            chat_history=self.chat_history,
        )

        # 3. Stream response from LLM
        full_response = ""
        try:
            with httpx.stream(
                "POST",
                self.api_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": MAX_TOKENS,
                    },
                },
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        import json
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            full_response += token
                            yield full_response, sources
                        if data.get("done", False):
                            break
                    except Exception:
                        continue

        except httpx.TimeoutException:
            error_msg = "⚠️ Timeout: il modello sta impiegando troppo tempo. Riprova con una domanda più breve."
            full_response += error_msg
            yield full_response, sources
        except httpx.HTTPStatusError as e:
            error_msg = f"⚠️ Errore comunicazione con Ollama: {e.response.status_code}"
            full_response += error_msg
            yield full_response, sources
        except Exception as e:
            error_msg = f"⚠️ Errore: {str(e)}"
            full_response += error_msg
            yield full_response, sources

        # 4. Update chat history
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": full_response})

        # Keep history bounded
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-12:]

    def ask_sync(self, question: str, top_k: int = 8) -> tuple[str, list[dict]]:
        """Non-streaming version of ask(). Returns final (response, sources)."""
        result = ("", [])
        for partial, sources in self.ask(question, top_k):
            result = (partial, sources)
        return result

    def clear_history(self):
        """Reset chat history."""
        self.chat_history = []

    def is_model_available(self) -> bool:
        """Check if the configured chat model is available on Ollama."""
        try:
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            model_base = self.model.split(":")[0]
            return any(model_base in m for m in models)
        except Exception:
            return False
