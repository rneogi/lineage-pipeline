# Data Lineage Pipeline v5.0

Extract and analyze data lineage from Sybase stored procedures with RAG-based querying.

## Quick Start

### Windows
```batch
git clone https://github.com/YOUR_USERNAME/lineage-pipeline.git
cd lineage-pipeline
build.bat setup
# Add API keys to venv\Scripts\activate.bat
build.bat run
```

### Linux/Mac
```bash
git clone https://github.com/YOUR_USERNAME/lineage-pipeline.git
cd lineage-pipeline
chmod +x build.sh
./build.sh setup
# Export API keys
./build.sh run
```

## Required API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| OpenAI | Embeddings (text-embedding-3-small) | https://platform.openai.com |
| ZeroEntropy | Reranking (zerank-2) | https://zeroentropy.dev |
| Anthropic | LLM (Claude) | https://console.anthropic.com |

## Project Structure

```
lineage-pipeline/
├── source_code/           # Python source files
│   ├── demo.py            # Main pipeline
│   ├── chat_interface.py  # Interactive chat
│   └── *.py               # Agent modules
├── input_data/            # SP/DDL files, Imperva logs
├── .vscode/               # VSCode settings
├── requirements.txt       # Python dependencies
├── build.bat              # Windows build script
├── build.sh               # Linux/Mac build script
└── README.md
```

## Build Commands

| Command | Description |
|---------|-------------|
| `build setup` | Create venv + install dependencies |
| `build run` | Run the lineage pipeline |
| `build rebuild` | Clean and regenerate ChromaDB |
| `build clean` | Remove all generated files |

## Pipeline Phases

1. **Parse** - Extract DDL catalog + stored procedure lineage
2. **Abstract** - Build YAML Knowledge Graph
3. **Retrieve** - RAG with ChromaDB + ZeroEntropy rerank
4. **Generate** - Excel/CSV lineage reports
5. **Validate** - Compare against ground truth

## Chat Interface

```batch
run_chat.bat
```

Query examples:
- "Which procedures write to tlprdets?"
- "Show attribute lineage for reserve_amt"
- "What tables does cl_cr_rate_main read from?"
