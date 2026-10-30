@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo TenantEase Predictive Intelligence
 echo ============================================
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    where py >nul 2>nul
    if %errorlevel%==0 (
        py -3.11 -m venv .venv 2>nul
        if errorlevel 1 py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo Could not create the virtual environment. Install Python 3.11 or newer and try again.
    pause
    exit /b 1
)
echo Installing / checking dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
)
echo Starting TenantEase in your browser...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
