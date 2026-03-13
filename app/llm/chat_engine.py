"""
ChatEngine — RAG-based chat using Ollama LLM.
Supports standard retrieval + map-reduce for aggregation queries.
"""

import json
import logging
from typing import Generator

import httpx

from app.config import (
    OLLAMA_HOST, TEMPERATURE, MAX_TOKENS, MAX_TOKENS_AGGREGATION,
    TOP_K, TOP_K_AGGREGATION, UserConfig,
)
from app.search.retriever import Retriever, detect_query_type
from app.llm.prompts import build_messages, build_map_prompt, build_reduce_prompt

logger = logging.getLogger(__name__)

# Maximum characters of context per LLM call (stay within model context window)
MAX_CONTEXT_CHARS = 12_000
# Batch size for map-reduce: how many files per MAP call
MAP_BATCH_FILES = 5


class ChatEngine:
    """RAG chat engine with hybrid search and map-reduce aggregation."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.chat_history: list[dict] = []

    @property
    def api_url(self) -> str:
        from app.config import OLLAMA_HOST
        return f"{OLLAMA_HOST}/api/chat"

    @property
    def model(self) -> str:
        return UserConfig.load().chat_model

    # ─── Main entry point ─────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k: int | None = None,
    ) -> Generator[tuple[str, list[dict]], None, None]:
        """
        Ask a question with RAG. Detects query type and routes accordingly.

        Yields:
            (partial_response_text, sources) tuples
        """
        query_type = detect_query_type(question)
        logger.info(f"Query type detected: {query_type} for: {question[:80]}...")

        if query_type == "aggregation":
            yield from self._ask_map_reduce(question)
        else:
            yield from self._ask_standard(question, top_k, query_type)

    # ─── Standard RAG (hybrid search) ─────────────────────────────────────

    def _ask_standard(
        self,
        question: str,
        top_k: int | None = None,
        query_type: str = "normal",
    ) -> Generator[tuple[str, list[dict]], None, None]:
        """Standard RAG: retrieve top-K → build prompt → stream answer."""

        if top_k is None:
            top_k = TOP_K_AGGREGATION if query_type == "extraction" else TOP_K

        # 1. Retrieve relevant chunks (hybrid: semantic + BM25)
        results = self.retriever.retrieve(question, top_k=top_k, query_type=query_type)
        context = self.retriever.format_context(results)
        sources = self.retriever.format_sources(results)

        # 2. Build prompt messages
        messages = build_messages(
            user_question=question,
            context=context,
            chat_history=self.chat_history,
        )

        max_tokens = MAX_TOKENS_AGGREGATION if query_type == "extraction" else MAX_TOKENS

        # 3. Stream response
        full_response = ""
        for token in self._stream_llm(messages, max_tokens=max_tokens):
            full_response += token
            yield full_response, sources

        # 4. Update chat history
        self._add_to_history(question, full_response)

    # ─── Map-Reduce (aggregation across ALL documents) ─────────────────────

    def _ask_map_reduce(
        self,
        question: str,
    ) -> Generator[tuple[str, list[dict]], None, None]:
        """
        Map-Reduce pattern for aggregation queries.

        MAP:  For each batch of files, ask LLM to extract the requested data.
        REDUCE: Merge all partial results into one comprehensive answer.
        """
        # 1. Get ALL chunks grouped by file
        chunks_by_file = self.retriever.retrieve_all_chunks()
        total_files = len(chunks_by_file)

        if total_files == 0:
            yield "⚠️ Nessun documento indicizzato.", []
            return

        # Show progress to user
        yield f"🔄 **Analisi completa di {total_files} file in corso...**\n\n_Modalità map-reduce attiva per coprire tutti i documenti._\n\n", []

        # 2. MAP phase: process files in batches
        partial_results = []
        files_processed = 0
        all_sources = []

        file_items = list(chunks_by_file.items())

        for batch_start in range(0, len(file_items), MAP_BATCH_FILES):
            batch = file_items[batch_start:batch_start + MAP_BATCH_FILES]

            # Build context from this batch of files
            context_parts = []
            batch_sources = []
            char_count = 0

            for filepath, chunks in batch:
                # Sort chunks by page/index for coherent reading
                chunks.sort(key=lambda c: (
                    c.get("metadata", {}).get("page", 0),
                    c.get("metadata", {}).get("chunk_index", 0),
                ))

                for chunk in chunks:
                    meta = chunk.get("metadata", {})
                    filename = meta.get("filename", "?")
                    page = meta.get("page", "")
                    page_str = f", pag. {page}" if page else ""
                    chunk_text = f"[{filename}{page_str}]\n{chunk['text']}"

                    # Don't exceed context budget
                    if char_count + len(chunk_text) > MAX_CONTEXT_CHARS:
                        break
                    context_parts.append(chunk_text)
                    char_count += len(chunk_text)

                    if not any(s["filepath"] == filepath for s in batch_sources):
                        batch_sources.append({
                            "filename": meta.get("filename", "?"),
                            "filepath": filepath,
                            "page": meta.get("page", ""),
                            "score": 1.0,
                            "preview": chunk["text"][:150] + "...",
                            "extraction": meta.get("extraction", "native"),
                        })

            if not context_parts:
                files_processed += len(batch)
                continue

            batch_context = "\n\n---\n\n".join(context_parts)

            # MAP call (synchronous, non-streaming)
            map_prompt = build_map_prompt(batch_context)
            messages = [
                {"role": "system", "content": map_prompt},
                {"role": "user", "content": question},
            ]

            map_result = self._call_llm_sync(messages, max_tokens=MAX_TOKENS)
            if map_result.strip():
                partial_results.append(map_result)

            all_sources.extend(batch_sources)
            files_processed += len(batch)

            progress_pct = int(files_processed / total_files * 100)
            yield (
                f"🔄 **Analisi in corso... {progress_pct}%** ({files_processed}/{total_files} file)\n\n"
                f"_Risultati parziali raccolti: {len(partial_results)} batch elaborati._\n\n"
            ), []

        # 3. REDUCE phase: merge all partial results
        if not partial_results:
            yield "⚠️ Non ho trovato questa informazione nei documenti indicizzati.", all_sources
            self._add_to_history(question, "Non ho trovato questa informazione.")
            return

        yield (
            f"📊 **Aggregazione risultati da {total_files} file...**\n\n"
        ), []

        combined_partials = "\n\n".join(partial_results)

        # If partials are small enough, send them all to REDUCE
        reduce_prompt = build_reduce_prompt(question, combined_partials)
        messages = [
            {"role": "system", "content": reduce_prompt},
            {"role": "user", "content": f"Unisci i risultati e rispondi alla domanda: {question}"},
        ]

        # Stream the final REDUCE response
        full_response = ""
        for token in self._stream_llm(messages, max_tokens=MAX_TOKENS_AGGREGATION):
            full_response += token
            yield full_response, all_sources

        self._add_to_history(question, full_response)

    # ─── LLM communication ────────────────────────────────────────────────

    def _stream_llm(
        self,
        messages: list[dict],
        max_tokens: int = MAX_TOKENS,
    ) -> Generator[str, None, None]:
        """Stream tokens from Ollama. Yields individual tokens."""
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
                        "num_predict": max_tokens,
                    },
                },
                timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except Exception:
                        continue

        except httpx.TimeoutException:
            yield "\n\n⚠️ Timeout: il modello sta impiegando troppo tempo. Riprova con una domanda più breve."
        except httpx.HTTPStatusError as e:
            yield f"\n\n⚠️ Errore comunicazione con Ollama: {e.response.status_code}"
        except Exception as e:
            yield f"\n\n⚠️ Errore: {str(e)}"

    def _call_llm_sync(
        self,
        messages: list[dict],
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Non-streaming LLM call. Returns full response text."""
        try:
            response = httpx.post(
                self.api_url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE,
                        "num_predict": max_tokens,
                    },
                },
                timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=10.0),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Sync LLM call failed: {e}")
            return ""

    # ─── History management ───────────────────────────────────────────────

    def _add_to_history(self, question: str, response: str):
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": response})
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-12:]

    def ask_sync(self, question: str, top_k: int | None = None) -> tuple[str, list[dict]]:
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
            from app.config import OLLAMA_HOST
            r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return any(m == self.model or m == f"{self.model}:latest" or (":" not in self.model and m.startswith(f"{self.model}:")) for m in models)
        except Exception:
            return False
