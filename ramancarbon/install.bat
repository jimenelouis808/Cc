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
REM La bateria completa son ~620 pruebas y unos seis minutos, casi todo
REM refinamientos Rietveld. Para comprobar una instalacion basta con que
REM los datos esten, la GUI este cableada y los cuatro instrumentos
REM arranquen de verdad.
python -m pytest ramancarbon\tests -q -k "packaging or gui_wiring or database or spectrum or io or echem"
if errorlevel 1 (
    echo.
    echo === Las pruebas han fallado. Revisa la salida de arriba. ===
    pause
    exit /b 1
)

echo.
echo Probando los cuatro instrumentos...
set "PRUEBA=%TEMP%\ramancarbon_check"
if exist "%PRUEBA%" rmdir /s /q "%PRUEBA%"
python -m ramancarbon.cli.main demo "%PRUEBA%\raman" >nul 2>&1 ^
 && python -m ramancarbon.cli.main analizar "%PRUEBA%\raman\demo_MWCNT_532nm.txt" --laser 532 --breve >nul 2>&1 ^
 && python -m ramancarbon.cli.main demo "%PRUEBA%\tmd" --tmd >nul 2>&1 ^
 && python -m ramancarbon.cli.main tmd "%PRUEBA%\tmd\demo_MoS2_1capa_532nm.txt" >nul 2>&1 ^
 && python -m ramancarbon.cli.main demo-datos drx "%PRUEBA%\drx" >nul 2>&1 ^
 && python -m ramancarbon.cli.main drx "%PRUEBA%\drx\demo_drx_CNT_FeSe.xye" --sin-refinar --breve >nul 2>&1 ^
 && python -m ramancarbon.cli.main demo-datos echem "%PRUEBA%\ec" >nul 2>&1 ^
 && python -m ramancarbon.cli.main echem --cv "%PRUEBA%\ec\demo_cv_condensador_20mVs.txt" --masa 2 --area 1 --breve >nul 2>&1
if errorlevel 1 (
    echo.
    echo === Algun instrumento no arranca. ===
    pause
    exit /b 1
)
rmdir /s /q "%PRUEBA%"
echo   Raman de carbono, Raman de TMD, difraccion y electroquimica: OK

echo.
echo === OK ===
echo.
echo Para usarlo, activa el entorno en cada terminal nueva:
echo.
echo     .venv\Scripts\activate
echo.
echo Y luego:
echo.
echo     ramancarbon-gui                          ^(la suite entera^)
echo.
echo     ramancarbon demo datos_prueba\
echo     ramancarbon demo-datos drx datos_drx\
echo     ramancarbon demo-datos echem datos_ec\
echo.
echo La bateria completa de pruebas ^(unos seis minutos^):
echo.
echo     pytest ramancarbon\tests -q
echo.
pause
