#!/usr/bin/env bash
# Instalador para Linux y macOS.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== ramancarbon: instalación ==="
echo

PY=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: hace falta Python 3.10 o más nuevo y no se ha encontrado."
    echo "Instálalo desde https://www.python.org/downloads/ y vuelve a intentarlo."
    exit 1
fi
echo "Usando $($PY --version) en $(command -v $PY)"

if [ ! -d .venv ]; then
    echo "Creando el entorno virtual en .venv…"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Instalando…"
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[dev]"

echo
if python -c 'import tkinter' 2>/dev/null; then
    echo "Tkinter: presente. La interfaz gráfica funcionará."
else
    echo "AVISO: falta Tkinter, así que la ventana no se abrirá."
    echo "  Ubuntu / Debian:  sudo apt install python3-tk"
    echo "  Fedora:           sudo dnf install python3-tkinter"
    echo "  Arch:             sudo pacman -S tk"
    echo "  macOS:            reinstala Python desde python.org"
    echo "La línea de comandos funciona igualmente."
fi

echo
echo "Ejecutando las pruebas…"
if python -m pytest ramancarbon/tests -q; then
    echo
    echo "=== OK ==="
else
    echo
    echo "=== Las pruebas han fallado. Revisa la salida de arriba. ==="
    exit 1
fi

cat <<'MSG'

Para usarlo, activa el entorno en cada terminal nueva:

    source .venv/bin/activate

Y luego:

    ramancarbon demo datos_prueba/
    ramancarbon analizar datos_prueba/demo_DWCNT_532nm.txt --laser 532
    ramancarbon-gui

MSG
