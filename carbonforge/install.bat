@echo off
REM carbonforge - installer for Windows.
REM Creates a .venv virtual environment, installs the package and checks Tkinter.
REM Safe to re-run. Double-click it or run it from cmd / PowerShell.

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==^> Buscando Python 3.10 o superior...

REM The py launcher ships with the python.org installer and is the most
REM reliable way to find a specific version on Windows.
set PYTHON=
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYTHON (
        py -%%V -c "import sys" >nul 2>&1 && set PYTHON=py -%%V
    )
)

REM Fall back to whatever "python" resolves to, if it is new enough.
if not defined PYTHON (
    python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set PYTHON=python
)

if not defined PYTHON (
    echo.
    echo   xx  No se encontro Python 3.10 o superior.
    echo.
    echo   Descargalo de https://www.python.org/downloads/
    echo   IMPORTANTE durante la instalacion:
    echo     - marca "Add Python to PATH"
    echo     - deja marcado "tcl/tk and IDLE"  ^(dibuja la ventana^)
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%o in ('%PYTHON% --version') do echo   ok  %%o

REM --- virtual environment ---------------------------------------------------
if not exist .venv (
    echo ==^> Creando entorno virtual en .venv
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo   xx  No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
) else (
    echo ==^> Reutilizando el entorno virtual existente ^(.venv^)
)

call .venv\Scripts\activate.bat
echo   ok  Entorno activado

REM --- install ---------------------------------------------------------------
echo ==^> Instalando carbonforge y sus dependencias
python -m pip install --upgrade pip --quiet
python -m pip install -e ".[dev]" --quiet
if errorlevel 1 (
    echo   xx  Fallo la instalacion. Revisa los mensajes anteriores.
    pause
    exit /b 1
)
echo   ok  carbonforge instalado

REM --- Tkinter check ---------------------------------------------------------
python -c "import tkinter" >nul 2>&1
if errorlevel 1 (
    echo   !!  Tkinter NO esta disponible: la interfaz grafica no arrancara.
    echo       Reinstala Python desde python.org marcando "tcl/tk and IDLE".
    echo       La linea de comandos si funciona.
) else (
    echo   ok  Tkinter disponible: la interfaz grafica funcionara
)

REM --- smoke test ------------------------------------------------------------
echo ==^> Comprobando la instalacion
python -m pytest -q >nul 2>&1
if errorlevel 1 (
    echo   !!  Algun test fallo. Ejecuta "pytest -q" para ver el detalle.
) else (
    echo   ok  Tests superados
)

echo.
echo Listo. Para usarlo, activa el entorno en cada terminal nueva:
echo.
echo     .venv\Scripts\activate
echo.
echo Y despues:
echo.
echo     carbonforge-gui        (interfaz grafica)
echo     carbonforge --help     (linea de comandos)
echo.
pause
