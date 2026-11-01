#!/usr/bin/env bash
# Arranca la interfaz de nanocarbon_biblio. Doble clic, o ./run_gui.sh
# Toda la lógica vive en start.py para que haya un solo sitio que mantener.
cd "$(dirname "$0")" || exit 1
for py in python3 python py; do
  if command -v "$py" >/dev/null 2>&1; then exec "$py" start.py "$@"; fi
done
echo "No encuentro Python. Instálalo desde https://www.python.org/downloads/"
echo "En macOS también vale:  brew install python@3.12"
read -r -p "Pulsa Enter para cerrar..."
exit 1
