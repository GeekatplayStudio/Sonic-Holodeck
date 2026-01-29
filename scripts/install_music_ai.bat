@echo off
SETLOCAL EnableDelayedExpansion
SET "COMFY_PATH=..\..\.."
SET "NODES_PATH=%COMFY_PATH%\custom_nodes"
SET "PYTHON_EXE=%COMFY_PATH%\python_embeded\python.exe"

IF NOT EXIST "%NODES_PATH%" (
    ECHO [ERROR] Could not find ComfyUI custom_nodes folder at: %NODES_PATH%
    PAUSE
    EXIT /B
)

CD /D "%NODES_PATH%"

:: 1. HEARTMULA (Vocals)
IF NOT EXIST "ComfyUI_FL-HeartMuLa" (
    git clone https://github.com/filliptm/ComfyUI_FL-HeartMuLa
)
ECHO Installing HeartMuLa dependencies (including torchtune, torchao, vector-quantize, modelscope)...
"%PYTHON_EXE%" -m pip install -r ComfyUI_FL-HeartMuLa\requirements.txt
"%PYTHON_EXE%" -m pip install torchao torchtune vector-quantize-pytorch modelscope

:: 2. AUDIOCRAFT (Native in Sonic-Holodeck now)
:: IF NOT EXIST "audio_diffusion_studio" (
::    git clone https://github.com/gitmylo/audio_diffusion_studio
::    "%PYTHON_EXE%" -m pip install -r audio_diffusion_studio\requirements.txt
:: )

:: 3. STABLE AUDIO (High Fidelity) - Repo currently unavailable/moved
:: IF NOT EXIST "ComfyUI-StableAudioOpen" (
::     git clone https://github.com/Stable-Audio-Tools/ComfyUI-StableAudioOpen
::     "%PYTHON_EXE%" -m pip install -r ComfyUI-StableAudioOpen\requirements.txt
:: )

:: 4. VRCH AUDIO (Mic Support)
IF NOT EXIST "comfyui-web-viewer" (
    git clone https://github.com/VrchStudio/comfyui-web-viewer
)

ECHO Installing dependencies for Web Viewer (Mic)...
"%PYTHON_EXE%" -m pip install -r comfyui-web-viewer\requirements.txt
:: Extra safeguard for deps that sometimes fail
"%PYTHON_EXE%" -m pip install pydub srt python-osc websockets

ECHO [SUCCESS] All Music AI nodes installed.
PAUSE