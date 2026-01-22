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
goto :INSTALL_DEPS_VENV

:INSTALL_DEPS
echo Using Python: !PYTHON_CMD!
echo.
echo ======================================================
echo NOTE: Installing directly into ComfyUI environment.
echo ======================================================
!PYTHON_CMD! -m pip install --upgrade pip setuptools wheel
!PYTHON_CMD! -m pip install -r requirements.txt --prefer-binary
!PYTHON_CMD! -m pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps

:FINISH
echo.
echo Installation Complete.
pause
exit /b 0

:INSTALL_DEPS_VENV
echo Using Virtual Environment Python: !PYTHON_CMD!
!PYTHON_CMD! -m pip install --upgrade pip setuptools wheel
!PYTHON_CMD! -m pip install -r requirements.txt --prefer-binary
!PYTHON_CMD! -m pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps
goto :FINISH

:INSTALL_DEPS_VENV
echo Using Venv...
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --prefer-binary
pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps
pip install soundfile librosa protobuf pesq pystoi torchmetrics torchdiffeq
goto :FINISH

:FINISH

echo.
echo ===================================================
echo INSTALLATION COMPLETE
echo To start, run ComfyUI with this venv active.
echo ===================================================

