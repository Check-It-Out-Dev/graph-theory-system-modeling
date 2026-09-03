@echo off
rem CodeMap launcher — embedded Python, no system requirements, no cloud.
rem Runs the same wizard a git clone runs: python codemap.py up
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
"%~dp0python-embed\python.exe" "%~dp0codemap.py" up %*
echo.
pause
