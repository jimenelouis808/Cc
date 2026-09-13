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
# La batería completa son ~620 pruebas y unos seis minutos, casi todo
# refinamientos Rietveld. Para comprobar una instalación no hace falta:
# lo que importa es que los datos estén, que la GUI esté cableada y que
# los cuatro instrumentos arranquen de verdad. La completa se lanza
# aparte, y el mensaje del final dice cómo.
echo "Comprobando la instalación…"
# Archivos concretos, no -k: "-k io" encaja tambien con "identificacion",
# "resolucion" y "orientacion", y lo que iba a ser medio minuto se
# convierte en cinco.
if ! python -m pytest -q \
        ramancarbon/tests/test_packaging.py \
        ramancarbon/tests/test_gui_wiring.py \
        ramancarbon/tests/test_database.py \
        ramancarbon/tests/test_spectrum.py \
        ramancarbon/tests/test_io.py \
        ramancarbon/tests/test_lineshapes.py; then
    echo
    echo "=== Las pruebas han fallado. Revisa la salida de arriba. ==="
    exit 1
fi

echo
echo "Probando los cuatro instrumentos…"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
{
    python -m ramancarbon.cli.main demo "$TMP/raman" \
        && python -m ramancarbon.cli.main analizar \
               "$TMP/raman/demo_MWCNT_532nm.txt" --laser 532 \
        && python -m ramancarbon.cli.main demo "$TMP/tmd" --tmd \
        && python -m ramancarbon.cli.main tmd \
               "$TMP/tmd/demo_MoS2_1capa_532nm.txt" \
        && python -m ramancarbon.cli.main demo-datos drx "$TMP/drx" \
        && python -m ramancarbon.cli.main drx \
               "$TMP/drx/demo_drx_CNT_FeSe.xye" --sin-refinar --breve \
        && python -m ramancarbon.cli.main demo-datos echem "$TMP/ec" \
        && python -m ramancarbon.cli.main echem \
               --cv "$TMP/ec/demo_cv_condensador_20mVs.txt" \
               --masa 2 --area 1 --breve \
        && python -m ramancarbon.cli.main figura \
               "$TMP/raman/demo_MWCNT_532nm.txt" \
               --salida "$TMP/figura.png" --preajuste acs \
        && python -m ramancarbon.cli.main exportar \
               "$TMP/raman/demo_MWCNT_532nm.txt" "$TMP/salida.jdx" \
        && python -m ramancarbon.cli.main proyecto crear \
               "$TMP/raman" "$TMP/sesion.rcproj"
} >/dev/null 2>&1 || {
    echo "=== Algún instrumento no arranca. Revisa la salida de arriba. ==="
    exit 1
}
echo "  Raman de carbono, Raman de TMD, difracción y electroquímica: OK"
echo "  Figuras, exportación y proyectos: OK"

echo
echo "=== OK ==="

cat <<'MSG'

Para usarlo, activa el entorno en cada terminal nueva:

    source .venv/bin/activate

Y luego:

    ramancarbon-gui                          # la suite entera

O por línea de comandos, un instrumento cada vez:

    ramancarbon demo datos_prueba/
    ramancarbon analizar datos_prueba/demo_DWCNT_532nm.txt --laser 532
    ramancarbon demo-datos drx datos_drx/
    ramancarbon drx datos_drx/demo_drx_CNT_FeSe.xye
    ramancarbon demo-datos echem datos_ec/
    ramancarbon echem --cv datos_ec/demo_cv_condensador_20mVs.txt --masa 2 --area 1
    ramancarbon figura datos_prueba/*.txt --salida figura.png --preajuste acs
    ramancarbon proyecto crear datos_prueba/ sesion.rcproj
    ramancarbon tiempos --rapido

La batería completa de pruebas (unos seis minutos, sobre todo
refinamientos Rietveld):

    pytest ramancarbon/tests -q

MSG
