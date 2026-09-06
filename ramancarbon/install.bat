@echo off
REM Instalador para Windows. Doble clic o ejecutar desde cmd.
setlocal
cd /d "%~dp0"

echo === ramancarbon: instalacion ===
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: no se encuentra Python.
    echo Instalalo desde https://www.python.org/downloads/ marcando
    echo   [x] Add Python to PATH
    echo   [x] tcl/tk and IDLE
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: hace falta Python 3.10 o mas nuevo.
    python --version
    pause
    exit /b 1
)

for /f "delims=" %%v in ('python --version') do echo Usando %%v

if not exist .venv (
    echo Creando el entorno virtual en .venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando...
python -m pip install --upgrade pip >nul
python -m pip install -e ".[dev]"
if errorlevel 1 (
    echo ERROR: la instalacion ha fallado.
    pause
    exit /b 1
)

echo.
python -c "import tkinter" 2>nul
if errorlevel 1 (
    echo AVISO: falta Tkinter, la ventana no se abrira.
    echo Reinstala Python desde python.org marcando "tcl/tk and IDLE".
    echo La linea de comandos funciona igualmente.
) else (
    echo Tkinter: presente. La interfaz grafica funcionara.
)

echo.
echo Ejecutando las pruebas...
python -m pytest ramancarbon\tests -q
if errorlevel 1 (
    echo.
    echo === Las pruebas han fallado. Revisa la salida de arriba. ===
    pause
    exit /b 1
)

echo.
echo === OK ===
echo.
echo Para usarlo, activa el entorno en cada terminal nueva:
echo.
echo     .venv\Scripts\activate
echo.
echo Y luego:
echo.
echo     ramancarbon demo datos_prueba\
echo     ramancarbon-gui
echo.
pause
