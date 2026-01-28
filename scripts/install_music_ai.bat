@echo off
SETLOCAL EnableDelayedExpansion
SET "COMFY_PATH=..\..\..\ComfyUI"
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
    "%PYTHON_EXE%" -m pip install -r ComfyUI_FL-HeartMuLa\requirements.txt
)

:: 2. AUDIOCRAFT (MusicGen/Humming)
IF NOT EXIST "comfyui-audiocraft" (
    git clone https://github.com/gitmylo/comfyui-audiocraft
    "%PYTHON_EXE%" -m pip install -r comfyui-audiocraft\requirements.txt
)

:: 3. STABLE AUDIO (High Fidelity)
IF NOT EXIST "ComfyUI-StableAudioOpen" (
    git clone https://github.com/Stable-Audio-Tools/ComfyUI-StableAudioOpen
    "%PYTHON_EXE%" -m pip install -r ComfyUI-StableAudioOpen\requirements.txt
)

:: 4. VRCH AUDIO (Mic Support)
IF NOT EXIST "comfyui-web-viewer" (
    git clone https://github.com/VrchStudio/comfyui-web-viewer
    "%PYTHON_EXE%" -m pip install -r comfyui-web-viewer\requirements.txt
)
ECHO [SUCCESS] All Music AI nodes installed.
PAUSE