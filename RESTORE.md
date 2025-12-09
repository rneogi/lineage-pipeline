# Restore Instructions

## Quick Start (New Machine)

```powershell
# 1. Clone the repo
git clone https://github.com/rneogi/lineage-pipeline.git
cd lineage-pipeline

# 2. Run setup (creates venv, installs dependencies)
build.bat setup

# 3. Add API keys to venv\Scripts\activate.bat
# Add these lines at the end of the file:
#   set OPENAI_API_KEY=your-openai-key
#   set ZEROENTROPY_API_KEY=your-zeroentropy-key
#   set ANTHROPIC_API_KEY=your-anthropic-key

# 4. Copy input_data folder from backup/original machine
# Required files:
#   - input_data/store-procedure-table.zip (or SPsample.zip)
#   - input_data/imperva_small.xlsx (or .zip)

# 5. Run the pipeline
build.bat run
```

## What Gets Restored from GitHub

| Item | Restored | Notes |
|------|----------|-------|
| source_code/*.py | Yes | All Python code |
| documentation/ | Yes | All docs |
| requirements.txt | Yes | Dependencies list |
| build.bat/build.sh | Yes | Build scripts |
| .vscode/ | Yes | VSCode settings |

## What You Must Restore Manually

| Item | How to Restore |
|------|----------------|
| venv/ | Run `build.bat setup` |
| API keys | Add to `venv\Scripts\activate.bat` |
| input_data/ | Copy from backup or original machine |
| chroma_db/ | Auto-rebuilt when you run `build.bat run` |
| output_artifacts/ | Auto-generated when you run pipeline |

## API Keys Required

| Service | Environment Variable | Get From |
|---------|---------------------|----------|
| OpenAI | OPENAI_API_KEY | https://platform.openai.com |
| ZeroEntropy | ZEROENTROPY_API_KEY | https://zeroentropy.dev |
| Anthropic | ANTHROPIC_API_KEY | https://console.anthropic.com |

## Build Commands

| Command | Description |
|---------|-------------|
| `build.bat setup` | Create venv + install dependencies |
| `build.bat run` | Run the lineage pipeline |
| `build.bat rebuild` | Clean ChromaDB and regenerate |
| `build.bat clean` | Remove all generated files |

## Troubleshooting

### ChromaDB install fails
```powershell
pip install chromadb --only-binary=:all:
```

### Unicode errors on Windows
```powershell
python -X utf8 source_code\demo.py
```

### Missing API keys error
Make sure you activated the venv first:
```powershell
venv\Scripts\activate
```
