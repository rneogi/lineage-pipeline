@echo off
REM ============================================================
REM Data Lineage Pipeline v5.0 - Windows Build Script
REM ============================================================
REM Usage: build.bat [setup|run|rebuild|clean]
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

if "%1"=="" goto :setup
if "%1"=="setup" goto :setup
if "%1"=="run" goto :run
if "%1"=="rebuild" goto :rebuild
if "%1"=="clean" goto :clean
if "%1"=="help" goto :help
goto :help

:setup
echo.
echo ============================================================
echo STEP 1: Creating virtual environment...
echo ============================================================
if not exist venv (
    python -m venv venv
    echo Created venv/
) else (
    echo venv/ already exists, skipping...
)

echo.
echo ============================================================
echo STEP 2: Installing dependencies...
echo ============================================================
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================================
echo STEP 3: Setting up API keys...
echo ============================================================
echo.
echo IMPORTANT: You need to add your API keys to venv\Scripts\activate.bat
echo Add these lines at the end of the file:
echo.
echo   set OPENAI_API_KEY=your-openai-key
echo   set ZEROENTROPY_API_KEY=your-zeroentropy-key
echo   set ANTHROPIC_API_KEY=your-anthropic-key
echo.
echo ============================================================
echo Setup complete! Next steps:
echo   1. Add API keys to venv\Scripts\activate.bat
echo   2. Run: build.bat run
echo ============================================================
goto :end

:run
echo.
echo ============================================================
echo Running pipeline...
echo ============================================================
call venv\Scripts\activate.bat
python -X utf8 source_code\demo.py
goto :end

:rebuild
echo.
echo ============================================================
echo Rebuilding ChromaDB and outputs...
echo ============================================================
if exist chroma_db (
    echo Removing old chroma_db/
    rmdir /s /q chroma_db
)
if exist output_artifacts (
    echo Removing old output_artifacts/
    rmdir /s /q output_artifacts
)
call venv\Scripts\activate.bat
python -X utf8 source_code\demo.py
goto :end

:clean
echo.
echo ============================================================
echo Cleaning generated files...
echo ============================================================
if exist venv (
    echo Removing venv/
    rmdir /s /q venv
)
if exist chroma_db (
    echo Removing chroma_db/
    rmdir /s /q chroma_db
)
if exist output_artifacts (
    echo Removing output_artifacts/
    rmdir /s /q output_artifacts
)
if exist intermediate (
    echo Removing intermediate/
    rmdir /s /q intermediate
)
if exist __pycache__ (
    rmdir /s /q __pycache__
)
echo Clean complete.
goto :end

:help
echo.
echo Usage: build.bat [command]
echo.
echo Commands:
echo   setup    - Create venv and install dependencies (default)
echo   run      - Run the lineage pipeline
echo   rebuild  - Clean ChromaDB and regenerate everything
echo   clean    - Remove all generated files (venv, outputs, etc.)
echo   help     - Show this help message
echo.
goto :end

:end
endlocal
