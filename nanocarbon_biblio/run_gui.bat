@echo off
REM Arranca la interfaz de nanocarbon_biblio. Doble clic sobre este fichero.
cd /d "%~dp0"
where py >nul 2>nul && (py start.py %* & goto :done)
where python >nul 2>nul && (python start.py %* & goto :done)
echo No encuentro Python. Instalalo desde https://www.python.org/downloads/
echo Marca la casilla "Add python.exe to PATH" durante la instalacion.
pause
:done
