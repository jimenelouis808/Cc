#!/usr/bin/env python3
"""Arrancador de nanocarbon_biblio: diagnostica, instala y lanza la interfaz.

Usa **solo la biblioteca estándar**, a propósito: tiene que poder ejecutarse en
un Python recién instalado, antes de que exista ningún entorno virtual y sin que
haya nada que instalar primero. Si este fichero necesitara pandas, no podría
decirte que falta pandas.

Uso::

    python3 start.py              # revisa, instala si hace falta, y abre la interfaz
    python3 start.py --check      # solo diagnostica, no instala ni lanza nada
    python3 start.py --demo       # genera además un corpus de prueba y lo deja listo
    python3 start.py --tests      # instala y ejecuta la batería de tests
    python3 start.py --reinstall  # recrea el entorno virtual desde cero

En Windows, `py start.py` o `python start.py`.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
APP = ROOT / "nanocarbon_biblio" / "app.py"
NEEDED = ("pandas", "streamlit", "rapidfuzz", "altair", "openpyxl")

OK, BAD, INFO, WARN = "  [OK]", "  [X] ", "  ->  ", "  [!] "


def say(mark: str, message: str) -> None:
    """Print one diagnostic line."""
    print(f"{mark} {message}", flush=True)


def fail(message: str, remedy: str = "") -> "NoReturn":  # type: ignore[valid-type]
    """Stop with a clear reason and, when there is one, the fix."""
    print()
    say(BAD, message)
    if remedy:
        print()
        print("  Cómo arreglarlo:")
        for line in remedy.strip().splitlines():
            print(f"    {line}")
    print()
    sys.exit(1)


def venv_python() -> Path:
    """Path of the interpreter inside the project's virtual environment."""
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check_python() -> None:
    """Refuse to continue on an interpreter the project cannot run on."""
    version = sys.version_info
    pretty = f"{version.major}.{version.minor}.{version.micro}"
    if version[:2] < MIN_PYTHON:
        fail(
            f"Python {pretty} es demasiado antiguo (hace falta "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} o superior).",
            "macOS:   brew install python@3.12\n"
            "Ubuntu:  sudo apt install python3.12 python3.12-venv\n"
            "Windows: instala Python 3.12 desde python.org y marca "
            "'Add python.exe to PATH'\n"
            "Después vuelve a ejecutar:  python3 start.py",
        )
    say(OK, f"Python {pretty} ({sys.executable})")


def check_layout() -> None:
    """Verify the script is sitting in a complete checkout."""
    missing = [p.name for p in (REQUIREMENTS, APP) if not p.exists()]
    if missing:
        fail(
            f"No encuentro {', '.join(missing)} junto a start.py.",
            "start.py tiene que estar dentro de la carpeta nanocarbon_biblio\n"
            "del repositorio. Comprueba dónde estás:\n"
            f"    cd {ROOT}\n"
            "    ls        # deberías ver requirements.txt, nanocarbon_biblio/, R/",
        )
    say(OK, f"Proyecto completo en {ROOT}")


def ensure_venv(reinstall: bool = False) -> Path:
    """Create the virtual environment if needed and return its interpreter.

    A virtual environment rather than a global install because a global pip on
    macOS and on modern Debian refuses to write to the system Python
    (``externally-managed-environment``), which is the single most common way
    this step fails.
    """
    if reinstall and VENV.exists():
        say(INFO, "Borrando el entorno virtual anterior…")
        shutil.rmtree(VENV)

    python = venv_python()
    if python.exists():
        say(OK, f"Entorno virtual en {VENV}")
        return python

    say(INFO, f"Creando el entorno virtual en {VENV} (una sola vez)…")
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 or not python.exists():
        detail = (result.stderr or result.stdout or "").strip()
        fail(
            "No pude crear el entorno virtual.\n" + (f"      {detail}" if detail else ""),
            "Ubuntu/Debian suele necesitar el paquete de venv aparte:\n"
            "    sudo apt install python3-venv\n"
            "Si estás en una carpeta sin permiso de escritura, muévela a tu home.",
        )
    say(OK, f"Entorno virtual creado en {VENV}")
    return python


def missing_packages(python: Path) -> list[str]:
    """Return the required packages that ``python`` cannot import."""
    probe = (
        "import importlib.util, sys\n"
        f"mods = {NEEDED!r}\n"
        "print(','.join(m for m in mods if importlib.util.find_spec(m) is None))"
    )
    result = subprocess.run(
        [str(python), "-c", probe], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return list(NEEDED)
    return [m for m in result.stdout.strip().split(",") if m]


def install(python: Path, *, extra: list[str] | None = None) -> None:
    """Install the project's requirements into the virtual environment."""
    say(INFO, "Instalando dependencias (puede tardar un par de minutos)…")
    command = [str(python), "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.run(command, capture_output=True, text=True, check=False)

    command = [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    if extra:
        command += extra
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-12:])
        fail(
            "La instalación de dependencias falló. Las últimas líneas de pip:\n\n"
            + "\n".join(f"      {line}" for line in tail.splitlines()),
            "Lo más habitual es que no haya salida a internet o que un proxy\n"
            "corporativo bloquee PyPI. Prueba:\n"
            f"    {python} -m pip install -r {REQUIREMENTS} -v\n"
            "Si usas proxy, exporta HTTPS_PROXY antes de volver a intentarlo.",
        )
    say(OK, "Dependencias instaladas")


def verify(python: Path) -> None:
    """Confirm the package itself imports inside the virtual environment."""
    result = subprocess.run(
        [str(python), "-c", "import nanocarbon_biblio, sys; print(nanocarbon_biblio.__version__)"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-6:]
        fail(
            "Las dependencias están, pero el paquete no importa:\n\n"
            + "\n".join(f"      {line}" for line in tail),
            "Suele significar que el checkout está incompleto. Prueba:\n"
            "    git status\n"
            "    git checkout claude/binance-scalping-bot-cut2op",
        )
    say(OK, f"nanocarbon_biblio {result.stdout.strip()} importa correctamente")


def free_port(preferred: int = 8501, attempts: int = 20) -> int:
    """Find a port Streamlit can bind to, starting at ``preferred``.

    The default port is often already taken by another Streamlit app, and its
    error message does not say so clearly.
    """
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def make_demo(python: Path) -> None:
    """Generate the synthetic corpus so the interface has something to show."""
    say(INFO, "Generando corpus de demostración…")
    result = subprocess.run(
        [str(python), "-m", "nanocarbon_biblio.cli", "demo",
         "--out", str(ROOT / "data" / "raw" / "demo"), "--n-works", "900"],
        cwd=str(ROOT), capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        say(WARN, "No pude generar el corpus de demostración; la interfaz abrirá vacía.")
        return
    say(OK, "Corpus de demostración en data/raw/demo")
    say(INFO, "En la pestaña 1, pulsa 'Cargar carpeta' para cargarlo.")


def run_tests(python: Path) -> int:
    """Run the test suite and return its exit code."""
    install(python, extra=["pytest"])
    print()
    return subprocess.run(
        [str(python), "-m", "pytest", "nanocarbon_biblio/tests", "-q"],
        cwd=str(ROOT), check=False,
    ).returncode


def launch(python: Path) -> "NoReturn":  # type: ignore[valid-type]
    """Start the Streamlit server in the foreground."""
    port = free_port()
    print()
    print("  " + "=" * 62)
    print(f"  Abriendo la interfaz en:   http://localhost:{port}")
    print("  Para cerrarla: Ctrl+C en esta ventana")
    print("  " + "=" * 62)
    print()
    command = [
        str(python), "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]
    try:
        sys.exit(subprocess.run(command, cwd=str(ROOT), check=False).returncode)
    except KeyboardInterrupt:
        print("\n  Interfaz cerrada.")
        sys.exit(0)


def main(argv: list[str] | None = None) -> int:
    """Diagnose, set up and launch. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Arranca la interfaz de nanocarbon_biblio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="solo diagnostica; no instala ni lanza nada")
    parser.add_argument("--demo", action="store_true",
                        help="genera además un corpus de prueba")
    parser.add_argument("--tests", action="store_true",
                        help="instala y ejecuta la batería de tests")
    parser.add_argument("--reinstall", action="store_true",
                        help="borra y recrea el entorno virtual")
    args = parser.parse_args(argv)

    print()
    print("  nanocarbon_biblio — revisión del entorno")
    print("  " + "-" * 62)
    check_python()
    check_layout()

    if args.check:
        python = venv_python()
        if not python.exists():
            say(WARN, f"Todavía no hay entorno virtual en {VENV}.")
            say(INFO, "Ejecuta `python3 start.py` (sin --check) para crearlo.")
            return 0
        say(OK, f"Entorno virtual en {VENV}")
        pending = missing_packages(python)
        if pending:
            say(WARN, f"Faltan paquetes: {', '.join(pending)}")
            say(INFO, "Ejecuta `python3 start.py` para instalarlos.")
            return 1
        say(OK, "Todas las dependencias están instaladas")
        verify(python)
        rscript = shutil.which("Rscript")
        say(OK if rscript else WARN,
            f"Rscript: {rscript}" if rscript
            else "Rscript no está en el PATH (solo hace falta para el paso de R)")
        print()
        say(INFO, "Todo listo. Arranca con:  python3 start.py")
        print()
        return 0

    python = ensure_venv(reinstall=args.reinstall)
    if args.tests:
        return run_tests(python)
    if missing_packages(python):
        install(python)
    else:
        say(OK, "Todas las dependencias están instaladas")
    verify(python)
    if args.demo:
        make_demo(python)
    launch(python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
