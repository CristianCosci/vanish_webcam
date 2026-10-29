@echo off
setlocal
cd /d "%~dp0\.."

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [VanishCam] Creo l'ambiente Python locale...
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto :error
)

"%VENV_PY%" -c "import cv2, mediapipe, numpy, pyvirtualcam, pygame" >nul 2>nul
if errorlevel 1 (
  echo [VanishCam] Installo i componenti necessari. Serve internet solo questa volta...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :error
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)

echo [VanishCam] Avvio la webcam...
"%VENV_PY%" vanish_cam.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Impossibile avviare VanishCam. Leggi README.md per le soluzioni comuni.
pause
exit /b 1
