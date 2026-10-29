@echo off
setlocal
cd /d "%~dp0\.."

where py >nul 2>nul
if errorlevel 1 (
  echo Installa Python 3 per Windows prima di creare l'eseguibile.
  goto :error
)

py -3 -c "import PyInstaller, cv2, mediapipe, numpy, pyvirtualcam, pygame" >nul 2>nul
if errorlevel 1 (
  echo Installo le dipendenze di build...
  py -3 -m pip install -r requirements.txt "pyinstaller>=6,<7"
  if errorlevel 1 goto :error
)

echo Scarico i modelli MediaPipe che mancano...
py -3 -c "from vanishcam.app import MODE_MODELS, download_models; [download_models(mode) for mode in MODE_MODELS]"
if errorlevel 1 goto :error

py -3 -m PyInstaller --noconfirm --distpath dist --workpath build windows\VanishCam.spec
if errorlevel 1 goto :error

"dist\VanishCam.exe" --self-test
if errorlevel 1 goto :error

echo.
echo Pronto: dist\VanishCam.exe
exit /b 0

:error
echo.
echo Build non riuscita. Controlla i messaggi qui sopra.
pause
exit /b 1
