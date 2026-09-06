#!/usr/bin/env bash
# carbonforge — installer for Linux and macOS.
#
# Creates a virtual environment in .venv, installs the package into it, and
# checks that Tkinter is available for the GUI. Safe to re-run.

set -euo pipefail

cd "$(dirname "$0")"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
info()  { printf '%s==>%s %s\n' "$BOLD" "$OFF" "$1"; }
ok()    { printf '%s  ok%s %s\n' "$GREEN" "$OFF" "$1"; }
warn()  { printf '%s  !!%s %s\n' "$YELLOW" "$OFF" "$1"; }
fail()  { printf '%s  xx%s %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }

# --- 1. locate a suitable Python -------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "No se encontró Python 3.10 o superior.
  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip python3-tk
  Fedora:        sudo dnf install python3 python3-tkinter
  macOS:         descarga desde https://www.python.org/downloads/"
fi
ok "Python: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

# --- 2. virtual environment -------------------------------------------------
if [ ! -d .venv ]; then
    info "Creando entorno virtual en .venv"
    "$PYTHON" -m venv .venv 2>/dev/null || fail "No se pudo crear el entorno.
  En Debian/Ubuntu falta el paquete venv: sudo apt install python3-venv"
else
    info "Reutilizando el entorno virtual existente (.venv)"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
ok "Entorno activado"

# --- 3. install -------------------------------------------------------------
info "Instalando carbonforge y sus dependencias"
python -m pip install --upgrade pip --quiet
python -m pip install -e ".[dev]" --quiet
ok "carbonforge instalado"

# --- 4. Tkinter check -------------------------------------------------------
# Tkinter ships with Python but is a separate OS package on most Linux
# distributions, and it cannot be installed with pip.
if python -c 'import tkinter' 2>/dev/null; then
    ok "Tkinter disponible: la interfaz gráfica funcionará"
else
    warn "Tkinter NO está instalado. La línea de comandos funciona igualmente,
       pero la interfaz gráfica necesita:
         Ubuntu/Debian: sudo apt install python3-tk
         Fedora:        sudo dnf install python3-tkinter
         Arch:          sudo pacman -S tk
       Después vuelve a ejecutar este script."
fi

# --- 5. smoke test ----------------------------------------------------------
info "Comprobando la instalación"
python -m pytest -q >/dev/null 2>&1 && ok "Tests superados" \
    || warn "Algún test falló. Ejecuta 'pytest -q' para ver el detalle."

cat <<EOF

${BOLD}Listo.${OFF} Para usarlo, activa el entorno en cada terminal nueva:

    source .venv/bin/activate

Y después:

    carbonforge-gui                      # interfaz gráfica
    carbonforge --help                   # línea de comandos

EOF
