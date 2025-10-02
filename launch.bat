@echo off
echo.
echo ====================================
echo  🌌 Astro-AI Platform Launcher 🌌
echo ====================================
echo.
echo Starting Galaxy Evolution Analysis Platform...
echo.

REM Check if streamlit is installed
streamlit --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Streamlit not found. Installing requirements...
    pip install -r requirements.txt
)

echo 🚀 Launching Astro-AI...
echo.
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the server when done.
echo.

streamlit run app.py

pause