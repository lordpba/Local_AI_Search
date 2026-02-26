"""
PrivateSearch — Main Gradio Application.
100% local document search with AI. Privacy-first.
"""

import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Generator

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import httpx

from app.config import (
    OLLAMA_HOST, OCR_MODEL, EMBEDDING_MODEL, PROFILES,
    ALL_EXTENSIONS, GRADIO_HOST, GRADIO_PORT,
    APP_TITLE, APP_SUBTITLE, UserConfig, DATA_DIR,
    host_to_container_path, container_to_host_path, HOST_HOME_PATH,
)
from app.indexer.document_loader import scan_folder, load_documents
from app.indexer.ocr_engine import OCREngine
from app.indexer.text_chunker import chunk_documents
from app.indexer.embedder import Embedder
from app.search.vector_store import VectorStore
from app.search.retriever import Retriever
from app.llm.chat_engine import ChatEngine
from app.ui.theme import create_theme, CUSTOM_CSS

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("PrivateSearch")

# ─── Global state ─────────────────────────────────────────────────────────────

vector_store: VectorStore | None = None
embedder: Embedder | None = None
retriever: Retriever | None = None
chat_engine: ChatEngine | None = None
ocr_engine: OCREngine | None = None


def init_components():
    """Initialize all components."""
    global vector_store, embedder, retriever, chat_engine, ocr_engine
    vector_store = VectorStore()
    embedder = Embedder()
    ocr_engine = OCREngine()
    retriever = Retriever(vector_store, embedder)
    chat_engine = ChatEngine(retriever)


# ─── Ollama helpers ───────────────────────────────────────────────────────────

def check_ollama_connection() -> tuple[bool, str]:
    """Check if Ollama is reachable."""
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        r.raise_for_status()
        return True, "Connesso"
    except Exception as e:
        return False, f"Non raggiungibile: {e}"


def get_installed_models() -> list[str]:
    """Get list of models installed in Ollama."""
    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def check_required_models(config: UserConfig) -> dict:
    """Check which required models are installed."""
    installed = get_installed_models()
    required = config.required_models
    status = {}
    for model in required:
        base = model.split(":")[0]
        status[model] = any(base in m for m in installed)
    return status


def pull_model(model_name: str) -> Generator:
    """Pull a model from Ollama with progress tracking."""
    try:
        with httpx.stream(
            "POST",
            f"{OLLAMA_HOST}/api/pull",
            json={"name": model_name, "stream": True},
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        status = data.get("status", "")
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        if total > 0:
                            pct = completed / total * 100
                            yield f"{status}: {pct:.0f}% ({completed // (1024*1024)}MB / {total // (1024*1024)}MB)"
                        else:
                            yield status
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"Errore: {e}"


# ─── UI Callbacks ─────────────────────────────────────────────────────────────

def on_check_system():
    """Check system status: Ollama connection + models."""
    connected, msg = check_ollama_connection()
    if not connected:
        return (
            f"🔴 Ollama non raggiungibile\n\n"
            f"Assicurati che Ollama sia installato e in esecuzione.\n"
            f"Scarica da: https://ollama.com\n\n"
            f"Errore: {msg}"
        )

    config = UserConfig.load()
    models_status = check_required_models(config)

    lines = ["🟢 Ollama connesso\n"]
    all_ok = True
    for model, installed in models_status.items():
        icon = "✅" if installed else "❌"
        lines.append(f"  {icon} {model}")
        if not installed:
            all_ok = False

    if all_ok:
        lines.append("\n✅ Tutti i modelli sono pronti!")
    else:
        lines.append("\n⚠️ Modelli mancanti. Clicca 'Scarica modelli' per installarli.")

    return "\n".join(lines)


def on_download_models(progress=gr.Progress()):
    """Download all required models."""
    config = UserConfig.load()
    models_status = check_required_models(config)
    missing = [m for m, installed in models_status.items() if not installed]

    if not missing:
        return "✅ Tutti i modelli sono già installati!"

    results = []
    for i, model in enumerate(missing):
        progress((i) / len(missing), f"Scaricamento {model}...")
        last_status = ""
        for status_msg in pull_model(model):
            last_status = status_msg
            progress((i) / len(missing), f"{model}: {status_msg}")
        results.append(f"✅ {model}: completato")
        progress((i + 1) / len(missing), f"{model} completato")

    config.models_downloaded = True
    config.save()

    return "\n".join(results) + "\n\n✅ Tutti i modelli scaricati! Puoi procedere."


def on_select_profile(profile_key: str):
    """Save selected profile."""
    config = UserConfig.load()
    config.profile = profile_key
    config.save()
    profile = PROFILES[profile_key]
    return f"Profilo selezionato: {profile['name']} ({profile['model']})"


def on_check_folder(folder_path: str):
    """Check folder and detect changes."""
    if not folder_path:
        return "❌ Inserisci un percorso.", ""

    # Translate host path → container path for Docker environments
    actual_path = host_to_container_path(folder_path.strip())
    logger.info(f"Folder check: user='{folder_path}' → actual='{actual_path}'")

    if not Path(actual_path).exists():
        msg = f"❌ Cartella non trovata: {folder_path}"
        if HOST_HOME_PATH:
            msg += f"\n\nℹ️ Il percorso deve essere dentro la tua home directory ({HOST_HOME_PATH})."
        return msg, ""

    if not Path(actual_path).is_dir():
        return "❌ Il percorso non è una cartella.", ""

    try:
        files = scan_folder(actual_path, ALL_EXTENSIONS)
    except Exception as e:
        return f"❌ Errore scansione: {e}", ""

    if not files:
        return "⚠️ Nessun file supportato trovato nella cartella.", ""

    # Save the user-facing (host) path for display
    config = UserConfig.load()
    config.folder_path = folder_path.strip()
    config.save()

    # Count by type
    from app.config import SUPPORTED_EXTENSIONS
    counts = {}
    for f in files:
        for category, exts in SUPPORTED_EXTENSIONS.items():
            if f["extension"] in exts:
                counts[category] = counts.get(category, 0) + 1
                break

    # Check vector store for changes
    init_components()
    changes = vector_store.detect_changes(files)

    summary = f"📂 {folder_path.strip()}\n\n"
    summary += f"📄 **{len(files)} file** trovati:\n"
    for cat, count in sorted(counts.items()):
        icons = {"text": "📝", "document": "📄", "image": "🖼️"}
        summary += f"  {icons.get(cat, '📎')} {cat}: {count}\n"

    total_size = sum(f["size"] for f in files)
    summary += f"\n💾 Dimensione totale: {total_size / (1024*1024):.1f} MB\n"

    changes_text = ""
    if changes["new"] or changes["modified"] or changes["deleted"]:
        changes_text = "⚠️ **Modifiche rilevate:**\n"
        if changes["new"]:
            changes_text += f"  🟢 {len(changes['new'])} file nuovi\n"
            for f in changes["new"][:5]:
                changes_text += f"    + {f['name']}\n"
            if len(changes["new"]) > 5:
                changes_text += f"    ... e altri {len(changes['new']) - 5}\n"
        if changes["modified"]:
            changes_text += f"  🟡 {len(changes['modified'])} file modificati\n"
            for f in changes["modified"][:5]:
                changes_text += f"    ~ {f['name']}\n"
        if changes["deleted"]:
            changes_text += f"  🔴 {len(changes['deleted'])} file rimossi\n"
            for fp in changes["deleted"][:5]:
                changes_text += f"    - {Path(fp).name}\n"
        changes_text += "\n**Clicca 'Indicizza' per aggiornare.**"
    elif not vector_store.is_empty:
        stats = vector_store.get_stats()
        changes_text = (
            f"✅ **Indice aggiornato** — nessuna modifica rilevata\n"
            f"  📊 {stats['total_chunks']} sezioni indicizzate da {stats['total_files']} file\n\n"
            f"Puoi iniziare a cercare!"
        )
    else:
        changes_text = "🆕 **Prima indicizzazione necessaria.** Clicca 'Indicizza documenti'."

    return summary, changes_text


def on_index_documents(folder_path: str, progress=gr.Progress()):
    """Run the full indexing pipeline: scan → load → OCR → chunk → embed → store."""
    actual_path = host_to_container_path(folder_path.strip()) if folder_path else ""
    if not actual_path or not Path(actual_path).is_dir():
        return "❌ Cartella non valida."

    init_components()

    # 1. Scan folder
    progress(0.0, "Scansione cartella...")
    files = scan_folder(actual_path, ALL_EXTENSIONS)
    if not files:
        return "⚠️ Nessun file supportato trovato."

    # 2. Detect changes (incremental)
    changes = vector_store.detect_changes(files)
    files_to_process = changes["new"] + changes["modified"]

    # 3. Handle deletions
    for deleted_path in changes["deleted"]:
        progress(0.05, f"Rimozione {Path(deleted_path).name}...")
        vector_store.remove_file(deleted_path)

    if not files_to_process and not changes["deleted"]:
        stats = vector_store.get_stats()
        return (
            f"✅ Nessuna modifica da processare.\n"
            f"📊 {stats['total_chunks']} sezioni da {stats['total_files']} file."
        )

    total_files = len(files_to_process)
    processed = 0
    total_chunks_added = 0

    # 4. Process each file
    for fi, file_info in enumerate(files_to_process):
        file_progress_base = 0.1 + (fi / total_files) * 0.85
        fp = Path(file_info["path"])

        progress(file_progress_base, f"[{fi+1}/{total_files}] Elaborazione {fp.name}...")

        # Load document
        docs, ocr_queue = load_documents(
            actual_path, [file_info],
            progress_callback=lambda p, m: progress(file_progress_base + p * 0.2 / total_files, m),
        )

        # OCR if needed
        if ocr_queue:
            progress(file_progress_base + 0.2 / total_files, f"OCR: {fp.name}...")
            ocr_docs = ocr_engine.process_ocr_queue(
                ocr_queue,
                progress_callback=lambda p, m: progress(
                    file_progress_base + (0.2 + p * 0.3) / total_files, m
                ),
            )
            docs.extend(ocr_docs)

        if not docs:
            logger.warning(f"No text extracted from {fp.name}")
            continue

        # Chunk
        progress(file_progress_base + 0.5 / total_files, f"Suddivisione: {fp.name}...")
        chunks = chunk_documents(docs)

        if not chunks:
            continue

        # Embed
        progress(file_progress_base + 0.6 / total_files, f"Embedding: {fp.name}...")
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_batch(
            texts,
            progress_callback=lambda p, m: progress(
                file_progress_base + (0.6 + p * 0.3) / total_files, m
            ),
        )

        # Store (upsert handles duplicates)
        progress(file_progress_base + 0.9 / total_files, f"Salvataggio: {fp.name}...")
        if file_info in changes["modified"]:
            vector_store.update_file(chunks, embeddings, file_info)
        else:
            vector_store.add_chunks(chunks, embeddings, file_info)

        processed += 1
        total_chunks_added += len(chunks)

    progress(1.0, "Indicizzazione completata!")

    stats = vector_store.get_stats()
    result = (
        f"✅ **Indicizzazione completata!**\n\n"
        f"📄 File elaborati: {processed}/{total_files}\n"
    )
    if changes["deleted"]:
        result += f"🗑️ File rimossi dall'indice: {len(changes['deleted'])}\n"
    result += (
        f"📊 Sezioni create: {total_chunks_added}\n"
        f"📊 Totale nell'indice: {stats['total_chunks']} sezioni da {stats['total_files']} file\n\n"
        f"Puoi iniziare a cercare nella scheda **💬 Chat**!"
    )
    return result


def on_chat_message(message: str, history: list):
    """Handle a chat message with RAG."""
    if not message or not message.strip():
        return history, ""

    # Check if index exists
    if vector_store is None or vector_store.is_empty:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": "⚠️ Nessun documento indicizzato. Vai nella scheda **📂 Documenti** e indicizza una cartella prima di cercare.",
        })
        return history, ""

    history.append({"role": "user", "content": message})

    # Stream response
    response_text = ""
    sources = []
    for partial, src in chat_engine.ask(message):
        response_text = partial
        sources = src

    # Format sources
    sources_text = ""
    if sources:
        sources_text = "\n\n---\n📋 **Fonti:**\n"
        for s in sources:
            page_str = f", pag. {s['page']}" if s['page'] else ""
            sources_text += f"- **{s['filename']}**{page_str} (rilevanza: {s['score']:.0%})\n"

    history.append({"role": "assistant", "content": response_text + sources_text})
    return history, ""


def on_clear_chat():
    """Clear chat history."""
    if chat_engine:
        chat_engine.clear_history()
    return [], ""


def on_clear_index():
    """Clear the entire vector store."""
    if vector_store:
        vector_store.clear()
    if ocr_engine:
        ocr_engine.clear_cache()
    return "🗑️ Indice e cache eliminati. Puoi re-indicizzare."


# ─── Build Gradio UI ──────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Build the complete Gradio application."""
    theme = create_theme()

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title=APP_TITLE,
        analytics_enabled=False,
    ) as app:

        # ─── Header ────────────────────────────────────────────
        gr.HTML(
            f"""
            <div style="text-align: center; padding: 20px 0 10px 0;">
                <h1 class="app-title">🔒 {APP_TITLE}</h1>
                <p class="app-subtitle">{APP_SUBTITLE}</p>
            </div>
            <div class="security-banner">
                <span class="lock-icon">🛡️</span>
                100% LOCALE — I tuoi dati non lasciano mai questo dispositivo.
                Nessuna connessione internet richiesta dopo la configurazione.
            </div>
            """
        )

        with gr.Tabs() as tabs:

            # ═══════════════ TAB 1: SETUP ═══════════════════════
            with gr.Tab("⚙️ Configurazione", id="setup"):
                gr.Markdown("### Configurazione iniziale")
                gr.Markdown(
                    "Verifica che Ollama sia installato e i modelli siano scaricati."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        system_status = gr.Textbox(
                            label="Stato del sistema",
                            lines=8,
                            interactive=False,
                            value="Clicca 'Verifica sistema' per controllare...",
                        )
                    with gr.Column(scale=1):
                        check_btn = gr.Button("🔍 Verifica sistema", variant="secondary")
                        download_btn = gr.Button("📥 Scarica modelli", variant="primary")

                gr.Markdown("### Profilo hardware")
                with gr.Row():
                    profile_radio = gr.Radio(
                        choices=[
                            ("⚡ Veloce — GPU 6-8 GB, risposte rapide", "fast"),
                            ("🎯 Preciso — GPU 12 GB+, risposte più accurate", "precise"),
                        ],
                        value=UserConfig.load().profile,
                        label="Seleziona il profilo adatto alla tua GPU",
                        interactive=True,
                    )
                profile_status = gr.Textbox(
                    label="",
                    interactive=False,
                    visible=True,
                    max_lines=1,
                )

                check_btn.click(fn=on_check_system, outputs=system_status)
                download_btn.click(fn=on_download_models, outputs=system_status)
                profile_radio.change(fn=on_select_profile, inputs=profile_radio, outputs=profile_status)

            # ═══════════════ TAB 2: DOCUMENTI ═══════════════════
            with gr.Tab("📂 Documenti", id="documents"):
                gr.Markdown("### Cartella documenti")
                gr.Markdown(
                    "Inserisci il percorso della cartella contenente i documenti da indicizzare."
                )

                with gr.Row():
                    folder_input = gr.Textbox(
                        label="Percorso cartella",
                        placeholder="/percorso/alla/cartella/documenti",
                        value=UserConfig.load().folder_path,
                        scale=3,
                    )
                    check_folder_btn = gr.Button("🔍 Controlla", variant="secondary", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        folder_info = gr.Markdown("Inserisci un percorso e clicca 'Controlla'.")
                    with gr.Column(scale=1):
                        changes_info = gr.Markdown("")

                index_btn = gr.Button(
                    "🚀 Indicizza documenti",
                    variant="primary",
                    size="lg",
                )
                index_result = gr.Markdown("")

                with gr.Accordion("🛠️ Strumenti avanzati", open=False):
                    clear_index_btn = gr.Button("🗑️ Elimina indice e cache", variant="stop")
                    clear_result = gr.Textbox(label="", interactive=False)

                check_folder_btn.click(
                    fn=on_check_folder,
                    inputs=folder_input,
                    outputs=[folder_info, changes_info],
                )
                index_btn.click(
                    fn=on_index_documents,
                    inputs=folder_input,
                    outputs=index_result,
                )
                clear_index_btn.click(
                    fn=on_clear_index,
                    outputs=clear_result,
                )

            # ═══════════════ TAB 3: CHAT ════════════════════════
            with gr.Tab("💬 Chat", id="chat"):
                chatbot = gr.Chatbot(
                    label="",
                    height=500,
                    type="messages",
                    show_copy_button=True,
                    placeholder=(
                        "🔒 **PrivateSearch**\n\n"
                        "Chiedimi qualsiasi cosa sui tuoi documenti.\n\n"
                        "_Esempi:_\n"
                        '- "Quali sono le analisi mediche di Mario del 2020?"\n'
                        '- "Elenca i documenti relativi alla proprietà in Italia"\n'
                        '- "Qual è la scadenza del contratto assicurativo?"'
                    ),
                )

                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Scrivi la tua domanda...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Invia", variant="primary", scale=1)

                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Nuova conversazione", variant="secondary", size="sm")

                # Chat events
                send_btn.click(
                    fn=on_chat_message,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                chat_input.submit(
                    fn=on_chat_message,
                    inputs=[chat_input, chatbot],
                    outputs=[chatbot, chat_input],
                )
                clear_chat_btn.click(
                    fn=on_clear_chat,
                    outputs=[chatbot, chat_input],
                )

        # ─── Footer ────────────────────────────────────────────
        gr.HTML(
            """
            <div class="privacy-footer">
                🔒 PrivateSearch v1.0 — Elaborazione 100% locale.
                Nessun dato viene trasmesso in rete.
                Tutti i tuoi documenti restano sul tuo dispositivo.
            </div>
            """
        )

    return app


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Launch the application."""
    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize components
    init_components()

    # Check folder on startup if configured
    config = UserConfig.load()
    actual_folder = host_to_container_path(config.folder_path) if config.folder_path else ""
    if actual_folder and Path(actual_folder).is_dir():
        logger.info(f"Configured folder: {config.folder_path} → {actual_folder}")
        files = scan_folder(actual_folder, ALL_EXTENSIONS)
        if files and vector_store.has_changes(files):
            logger.info("Changes detected in document folder")
        else:
            logger.info("No changes detected")

    # Build and launch app
    app = create_app()
    app.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=False,          # Never share — privacy first
        show_error=True,
        favicon_path=None,
    )


if __name__ == "__main__":
    main()
