@echo off
setlocal enabledelayedexpansion
echo Installing Geekatplay Studio's Sonic-Holodeck...

:: 1. Try to find ComfyUI Embedded Python (Best for Portable Builds)
if exist "..\..\..\python_embeded\python.exe" (
    echo Found ComfyUI Portable Python...
    set "PYTHON_CMD=..\..\..\python_embeded\python.exe"
    goto :INSTALL_DEPS
)

:: 2. Try to find System Python
python --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :CHECK_VENV
)

:: 3. Try py launcher
py --version >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py"
    goto :CHECK_VENV
)

echo Python not found. Please install Python or run this from a ComfyUI environment.
pause
exit /b 1

:CHECK_VENV
echo Using System Python: !PYTHON_CMD!
echo Checking for Virtual Environment...
if not exist venv (
    echo Creating venv...
    !PYTHON_CMD! -m venv venv
)
call venv\Scripts\activate.bat
set "PYTHON_CMD=python"
goto :INSTALL_DEPS

:INSTALL_DEPS
echo Using Python: !PYTHON_CMD!
echo.
echo ======================================================
echo NOTE: Installing dependencies...
echo ======================================================

:: Upgrade pip
!PYTHON_CMD! -m pip install --upgrade pip setuptools wheel

:: Install Generic Dependencies (Explicitly list them to ensure they install even if requirements.txt is empty)
!PYTHON_CMD! -m pip install soundfile librosa protobuf pesq pystoi torchmetrics torchdiffeq matplotlib einops numpy scipy

:: Install Safe Dependencies for Models (No heavy Torch deps)
!PYTHON_CMD! -m pip install datasets diffusers transformers accelerate sentencepiece huggingface_hub

:: Install Complex Audio Libraries via Git (No Deps to avoid Torch conflicts)
echo Installing Tangoflux...
!PYTHON_CMD! -m pip install tangoflux --no-deps

echo Installing Demucs...
!PYTHON_CMD! -m pip install git+https://github.com/facebookresearch/demucs.git@main#egg=demucs --no-deps

echo Installing Bark...
!PYTHON_CMD! -m pip install git+https://github.com/suno-ai/bark.git --no-deps

echo Installing Audiocraft...
!PYTHON_CMD! -m pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps

echo.
echo ===================================================
echo INSTALLATION COMPLETE
echo Please RESTART ComfyUI completely.
echo ===================================================
pause
exit /b 0

