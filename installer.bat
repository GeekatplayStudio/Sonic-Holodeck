@echo off
echo Installing Geekatplay Studio's Sonic-Holodeck...

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing core requirements...
pip install -r requirements.txt

echo.
echo ===================================================
echo INSTALLATION COMPLETE
echo To start, run ComfyUI with this venv active.
echo ===================================================
pause
