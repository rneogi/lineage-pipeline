#!/bin/bash
# ============================================================
# Data Lineage Pipeline v5.0 - Linux/Mac Build Script
# ============================================================
# Usage: ./build.sh [setup|run|rebuild|clean]
# ============================================================

set -e
cd "$(dirname "$0")"

setup() {
    echo ""
    echo "============================================================"
    echo "STEP 1: Creating virtual environment..."
    echo "============================================================"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "Created venv/"
    else
        echo "venv/ already exists, skipping..."
    fi

    echo ""
    echo "============================================================"
    echo "STEP 2: Installing dependencies..."
    echo "============================================================"
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    echo ""
    echo "============================================================"
    echo "STEP 3: Setting up API keys..."
    echo "============================================================"
    echo ""
    echo "IMPORTANT: Set your API keys as environment variables:"
    echo ""
    echo "  export OPENAI_API_KEY=your-openai-key"
    echo "  export ZEROENTROPY_API_KEY=your-zeroentropy-key"
    echo "  export ANTHROPIC_API_KEY=your-anthropic-key"
    echo ""
    echo "Or add them to venv/bin/activate"
    echo ""
    echo "============================================================"
    echo "Setup complete! Next steps:"
    echo "  1. Set API keys (export or add to activate)"
    echo "  2. Run: ./build.sh run"
    echo "============================================================"
}

run() {
    echo ""
    echo "============================================================"
    echo "Running pipeline..."
    echo "============================================================"
    source venv/bin/activate
    python source_code/demo.py
}

rebuild() {
    echo ""
    echo "============================================================"
    echo "Rebuilding ChromaDB and outputs..."
    echo "============================================================"
    [ -d "chroma_db" ] && rm -rf chroma_db && echo "Removed chroma_db/"
    [ -d "output_artifacts" ] && rm -rf output_artifacts && echo "Removed output_artifacts/"
    source venv/bin/activate
    python source_code/demo.py
}

clean() {
    echo ""
    echo "============================================================"
    echo "Cleaning generated files..."
    echo "============================================================"
    [ -d "venv" ] && rm -rf venv && echo "Removed venv/"
    [ -d "chroma_db" ] && rm -rf chroma_db && echo "Removed chroma_db/"
    [ -d "output_artifacts" ] && rm -rf output_artifacts && echo "Removed output_artifacts/"
    [ -d "intermediate" ] && rm -rf intermediate && echo "Removed intermediate/"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "Clean complete."
}

help() {
    echo ""
    echo "Usage: ./build.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup    - Create venv and install dependencies (default)"
    echo "  run      - Run the lineage pipeline"
    echo "  rebuild  - Clean ChromaDB and regenerate everything"
    echo "  clean    - Remove all generated files (venv, outputs, etc.)"
    echo "  help     - Show this help message"
    echo ""
}

# Main
case "${1:-setup}" in
    setup)   setup ;;
    run)     run ;;
    rebuild) rebuild ;;
    clean)   clean ;;
    help)    help ;;
    *)       help ;;
esac
