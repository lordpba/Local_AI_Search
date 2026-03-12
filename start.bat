@echo off
REM ─── PrivateSearch — Windows Launcher ─────────────────────
REM Avvia l'applicazione e apre il browser.
REM Prerequisiti: Docker Desktop + Ollama installati e in esecuzione.
REM ──────────────────────────────────────────────────────────

title PrivateSearch - Avvio
color 0A

echo.
echo  ========================================
echo   🔒 PrivateSearch — Avvio...
echo  ========================================
echo.

REM ─── Check Docker ───────────────────────────────────────
where docker >nul 2>&1
if errorlevel 1 (
    echo  ❌ Docker non trovato.
    echo     Installa Docker Desktop da: https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo  ❌ Docker non è in esecuzione.
    echo     Avvia Docker Desktop e riprova.
    echo.
    pause
    exit /b 1
)

echo  ✅ Docker OK

REM ─── Check Ollama ───────────────────────────────────────
set OLLAMA_OK=0
where ollama >nul 2>&1
if not errorlevel 1 (
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if not errorlevel 1 (
        set OLLAMA_OK=1
        echo  ✅ Ollama OK
    )
)

if "%OLLAMA_OK%"=="0" (
    echo  ⚠️  Ollama non raggiungibile su localhost:11434
    echo     Assicurati che Ollama sia installato e in esecuzione.
    echo     Installa da: https://ollama.com
    echo.
    echo     Continuo l'avvio...
)

REM ─── Generate .env ──────────────────────────────────────
REM On Windows Docker Desktop, file permissions work differently.
REM We don't need PUID/PGID — containers run with the correct user.
echo PUID=1000> "%~dp0docker\.env"
echo PGID=1000>> "%~dp0docker\.env"
echo HOST_HOME_PATH=%USERPROFILE:\=/%>> "%~dp0docker\.env"
echo OLLAMA_HOST=http://host.docker.internal:11434>> "%~dp0docker\.env"

REM ─── Build & Start ──────────────────────────────────────
echo.
echo  🚀 Avvio container...
docker compose -f "%~dp0docker\docker-compose.yml" up -d --build

REM ─── Wait for app ───────────────────────────────────────
echo.
set /a ATTEMPTS=0
:wait_loop
set /a ATTEMPTS+=1
if %ATTEMPTS% gtr 30 goto timeout
curl -s http://localhost:7860/ >nul 2>&1
if errorlevel 1 (
    echo|set /p="."
    timeout /t 2 /nobreak >nul
    goto wait_loop
)

echo.
echo  ✅ PrivateSearch è pronto!
goto open_browser

:timeout
echo.
echo  ⚠️ L'app sta ancora caricando, apro il browser...

:open_browser
echo.
echo  🌐 Apertura browser...
start http://localhost:7860

echo.
echo  ╔════════════════════════════════════════════════╗
echo  ║  🔒 PrivateSearch è in esecuzione              ║
echo  ║  📍 http://localhost:7860                      ║
echo  ║                                                ║
echo  ║  Per fermare:                                  ║
echo  ║    docker compose -f docker\docker-compose.yml ║
echo  ║    down                                        ║
echo  ╚════════════════════════════════════════════════╝
echo.
pause
