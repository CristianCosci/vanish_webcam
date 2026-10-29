@echo off
call "%~dp0avvia.bat" --virtual-camera %*
exit /b %errorlevel%
