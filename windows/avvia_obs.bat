@echo off
call "%~dp0avvia.bat" --no-virtual-camera %*
exit /b %errorlevel%
