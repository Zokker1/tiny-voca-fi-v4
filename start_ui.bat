@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virhe: .venv puuttuu. Asenna riippuvuudet ensin.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" server.py --open
pause
