#!/usr/bin/env bash
# ─── PrivateSearch — Launcher ──────────────────────────────
# Avvia l'applicazione e apre il browser.
# Prerequisiti: Docker Desktop + Ollama installati e in esecuzione.
# ────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/docker/.env"
APP_URL="http://localhost:7860"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${GREEN}🔒 PrivateSearch — Avvio...${NC}"
echo ""

# ─── Check Docker ─────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker non trovato.${NC}"
    echo "   Installa Docker Desktop da: https://www.docker.com/products/docker-desktop/"
    echo ""
    exit 1
fi

if ! docker info &> /dev/null 2>&1; then
    echo -e "${RED}❌ Docker non è in esecuzione.${NC}"
    echo "   Avvia Docker Desktop e riprova."
    echo ""
    exit 1
fi

echo -e "  ✅ Docker OK"

# ─── Check Ollama ─────────────────────────────────────────
OLLAMA_OK=false
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_OK=true
        echo -e "  ✅ Ollama OK"
    fi
fi

if [ "$OLLAMA_OK" = false ]; then
    echo -e "${YELLOW}⚠️  Ollama non raggiungibile su localhost:11434${NC}"
    echo "   Assicurati che Ollama sia installato e in esecuzione."
    echo "   Installa da: https://ollama.com"
    echo ""
    echo "   Continuo l'avvio... potrai configurare Ollama dall'app."
fi

# ─── Configure Ollama for Docker access ───────────────────
# Ollama must listen on 0.0.0.0 so the Docker container can reach it.
if [ "$OLLAMA_OK" = true ]; then
    OLLAMA_BIND=$(ss -tlnp 2>/dev/null | grep ":11434" | head -1 || true)
    if echo "$OLLAMA_BIND" | grep -q "127.0.0.1"; then
        echo ""
        echo -e "${YELLOW}⚠️  Ollama ascolta solo su localhost — necessario 0.0.0.0 per Docker.${NC}"
        echo "   Configuro automaticamente..."
        
        OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
        OVERRIDE_FILE="$OVERRIDE_DIR/docker-access.conf"
        
        if [[ -f /etc/systemd/system/ollama.service ]]; then
            sudo mkdir -p "$OVERRIDE_DIR"
            echo -e '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"' | sudo tee "$OVERRIDE_FILE" > /dev/null
            sudo systemctl daemon-reload
            sudo systemctl restart ollama
            sleep 3
            echo -e "  ✅ Ollama configurato su 0.0.0.0"
        else
            echo -e "${YELLOW}   Imposta manualmente: OLLAMA_HOST=0.0.0.0 ollama serve${NC}"
        fi
    else
        echo -e "  ✅ Ollama: bind su tutte le interfacce"
    fi
fi

# ─── Generate .env for Docker Compose ─────────────────────
echo "PUID=$(id -u)" > "$ENV_FILE"
echo "PGID=$(id -g)" >> "$ENV_FILE"
echo -e "  ✅ UID/GID: $(id -u):$(id -g)"

# ─── Build & Start ────────────────────────────────────────
echo ""
echo -e "  🚀 Avvio container..."
docker compose -f "$COMPOSE_FILE" up -d --build 2>&1 | tail -5

# ─── Wait for app to be ready ──────────────────────────────
echo ""
echo -n "  ⏳ In attesa che l'app sia pronta"
for i in $(seq 1 30); do
    if curl -s "$APP_URL" > /dev/null 2>&1; then
        echo ""
        echo -e "  ${GREEN}✅ PrivateSearch è pronto!${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# ─── Open browser ─────────────────────────────────────────
echo ""
echo -e "  🌐 Apertura browser su ${GREEN}${APP_URL}${NC}"
echo ""

# Cross-platform browser open
if command -v xdg-open &> /dev/null; then
    xdg-open "$APP_URL" 2>/dev/null &
elif command -v open &> /dev/null; then
    open "$APP_URL" 2>/dev/null &
elif command -v start &> /dev/null; then
    start "$APP_URL" 2>/dev/null &
fi

echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  🔒 PrivateSearch è in esecuzione              ║${NC}"
echo -e "${GREEN}║  📍 ${APP_URL}                    ║${NC}"
echo -e "${GREEN}║                                                ║${NC}"
echo -e "${GREEN}║  Per fermare: docker compose -f               ║${NC}"
echo -e "${GREEN}║    docker/docker-compose.yml down              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""
