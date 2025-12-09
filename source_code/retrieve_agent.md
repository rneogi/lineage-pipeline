# Retrieve Agent - PHASE 3 (v4.2)

## Overview

The Retrieve Agent builds a RAG (Retrieval-Augmented Generation) library from the YAML knowledge graph. It creates semantic embeddings for efficient lineage queries with telemetry-enhanced ranking and two-stage reranking.

## Usage

```bash
cd source_code
python retrieve_agent.py
```

**Prerequisite:** Run `abstract_agent.py` first to generate the input file.

## Input Files

| File | Location | Description |
|------|----------|-------------|
| `abstract_output.json` | `intermediate/` | Output from Abstract Agent (contains YAML graph) |

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `lineage_rag.pkl` | `output_artifacts/` | Binary RAG library with embeddings (~90 MB) |
| `retrieve_output.json` | `intermediate/` | JSON summary for next stage |

## RAG Document Types (v4.2)

The agent creates five types of documents for retrieval:

### 1. Catalog Documents
```json
{
  "type": "catalog",
  "table": "tclmdets",
  "columns": ["col1", "col2"],
  "text": "Table tclmdets with columns: col1, col2. Types: varchar, int",
  "telemetry_score": 0.05,
  "access_frequency": 100
}
```

### 2. Procedure Documents
```json
{
  "type": "procedure",
  "name": "AAA_BORDEREAU_DETAILS_LOAD",
  "source_tables": ["tclmdets", "tnumgen"],
  "target_tables": ["tlprdets"],
  "text": "Procedure AAA_BORDEREAU_DETAILS_LOAD reads from tclmdets, tnumgen and writes to tlprdets",
  "telemetry_score": 0.08,
  "call_frequency": 50
}
```

### 3. Table Lineage Documents
```json
{
  "type": "table_lineage",
  "source": "tclmdets",
  "target": "tlprdets",
  "procedure": "AAA_BORDEREAU_DETAILS_LOAD",
  "text": "Table Lineage: tclmdets -> tlprdets via AAA_BORDEREAU_DETAILS_LOAD",
  "telemetry_score": 0.012
}
```

### 4. Attribute Lineage Documents
```json
{
  "type": "attribute_lineage",
  "source_table": "tclmdets",
  "source_column": "clm_office_cd",
  "target_table": "tlprdets",
  "target_column": "clm_office_cd",
  "procedure": "AAA_BORDEREAU_DETAILS_LOAD",
  "confidence": 0.95,
  "mapping_type": "INSERT_SELECT",
  "text": "Attribute Lineage: tclmdets.clm_office_cd -> tlprdets.clm_office_cd via AAA_BORDEREAU_DETAILS_LOAD (confidence: 0.95, type: INSERT_SELECT)",
  "telemetry_score": 0.019
}
```

### 5. Parameter Lineage Documents (NEW in v4.2)
```json
{
  "type": "parameter_lineage",
  "parameter": "@clm_no",
  "table": "tclmdets",
  "column": "clm_no",
  "procedure": "AAA_BORDEREAU_DETAILS_LOAD",
  "pattern": "t.col = @p",
  "confidence": 0.9,
  "text": "Parameter @clm_no filters tclmdets.clm_no in AAA_BORDEREAU_DETAILS_LOAD (t.col = @p)",
  "telemetry_score": 0.03
}
```

## Embedding Models

The agent supports two embedding approaches:

### Semantic Embeddings (Preferred)
- Model: `all-MiniLM-L6-v2` from sentence-transformers
- Dimensions: 384
- Better semantic understanding
- Requires: `pip install sentence-transformers`

### TF-IDF Fallback
- Auto-activates if sentence-transformers unavailable
- Max features: 1000
- N-gram range: (1, 2)
- Fast and lightweight

## Two-Stage Retrieval (NEW in v4.2)

### Stage 1: Semantic Search (`search()`)
```python
results = rag.search(
    "show me frequently accessed tables",
    k=20,
    boost_telemetry=True,
    telemetry_weight=0.3
)
```

### Stage 2: Metadata Reranking (`query()`)
```python
results = rag.query(
    "how is tlprdets.amount computed",
    k=10,
    boost_telemetry=True
)
```

### Reranking Bonuses

The `query()` method applies metadata-based reranking:

| Query Pattern | Document Type | Bonus |
|--------------|---------------|-------|
| "how", "compute", "map" | `attribute_lineage` | +0.30 |
| "table", "flow", "lineage" | `table_lineage` | +0.25 |
| Contains "@" | `parameter_lineage` | +0.30 |
| Matches table hint | Any with matching table | +0.30 |
| Matches column hint | Any with matching column | +0.30 |
| Matches procedure hint | Any with matching proc | +0.20 |

### Query Hint Extraction

The `_extract_hints()` method parses queries for:
- **Table.column patterns**: `tclmdets.clm_no` → table=`tclmdets`, col=`clm_no`
- **Procedure patterns**: `*_prs`, `*_load`, `*_enq` suffixes

## Telemetry-Enhanced Retrieval

Query results are ranked using a combined score:

```
Stage 1 score = (1 - weight) * semantic_similarity + weight * telemetry_score
Stage 2 score = Stage 1 score + rerank_bonus
```

Default telemetry weight: 0.3 (30%)

### Result Structure
```json
{
  "type": "attribute_lineage",
  "table": "tclmdets",
  "text": "Attribute Lineage: ...",
  "score": 0.85,
  "score_stage1": 0.55,
  "semantic_score": 0.365,
  "telemetry_contribution": 0.185
}
```

## Performance

| Dataset Size | Documents | Build Time | RAG Size |
|-------------|-----------|------------|----------|
| Small (demo) | ~2,200 | ~25s | ~90 MB |
| Medium | ~5,000 | ~60s | ~200 MB |
| Large | ~10,000 | ~120s | ~400 MB |

## Next Stage

After Retrieve Agent completes, run the **Generate Agent** to create CSV/Excel reports:

```bash
python generate_agent.py
```
