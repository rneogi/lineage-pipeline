# Validate Agent - PHASE 5 (v4.2)

## Overview

The Validate Agent is the final stage of the lineage extraction pipeline. It validates the generated lineage against ground truth (if available) or performs self-validation, then generates compliance documentation.

## Usage

```bash
cd source_code
python validate_agent.py
```

**Prerequisite:** Run `generate_agent.py` first to generate the input file.

## Input Files

| File | Location | Description |
|------|----------|-------------|
| `generate_output.json` | `intermediate/` | Output from Generate Agent (contains YAML graph) |
| `annotated_groundtruth.yaml` | `input_data/` | Optional ground truth for validation |

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `validation_report.json` | `output_artifacts/` | Full validation report with metrics |
| `compliance_thresholds.json` | `documentation/` | Compliance thresholds and results (JSON) |
| `compliance_summary.txt` | `documentation/` | Human-readable compliance summary |
| `validate_output.json` | `intermediate/` | JSON summary for pipeline tracking |

## Validation Modes

### With Ground Truth
If `annotated_groundtruth.yaml` exists, validation compares generated lineage against it:
- Computes precision, recall, and F1 scores
- Identifies false positives (extra edges)
- Identifies false negatives (missing edges)

### Self-Validation (No Ground Truth)
Without ground truth, the agent performs sanity checks:
- Validates internal consistency
- Reports 100% precision/recall (comparing to itself)
- Still generates mapping quality analysis

## Metrics Computed

### Table Lineage
- **Precision**: Fraction of generated edges that are correct
- **Recall**: Fraction of ground truth edges that were found
- **F1 Score**: Harmonic mean of precision and recall

### Attribute Lineage
Same metrics, but for column-to-column mappings:
- `(source_table, source_column, target_table, target_column)`

### Mapping Quality
- Total number of attribute mappings
- Average confidence score
- Count of high-confidence mappings (>= 90%)
- Breakdown by mapping type

## Compliance Thresholds

The agent validates against these thresholds (configurable):

| Threshold | Default | Description |
|-----------|---------|-------------|
| Confidence Threshold | 90% | Minimum precision/recall to pass |
| High Confidence Mapping | 90% | Threshold for "high confidence" mappings |

### Mapping Type Confidence Levels

| Type | Confidence | Description |
|------|------------|-------------|
| `INSERT_SELECT` | 95% | Explicit INSERT...SELECT with column mapping |
| `UPDATE_SET` | 90% | Explicit UPDATE SET col = table.col |
| `UPDATE_SET_INFERRED` | 80% | UPDATE with inferred source column |
| `NAME_MATCH_HEURISTIC` | 70% | Fallback: matching column names |

## Output Formats

### validation_report.json

```json
{
  "validation_timestamp": "2025-11-22T...",
  "version": "4.2",
  "confidence_threshold": 0.90,
  "improvements": [
    "Deterministic attribute lineage (no cross-product)",
    "Parameter lineage extraction (@param -> table.column)",
    "Telemetry-enriched confidence scores",
    "Two-stage RAG with reranking"
  ],
  "table_lineage": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "tp": 720,
    "fp": 0,
    "fn": 0,
    "confident": true,
    "false_negatives": [],
    "false_positives": []
  },
  "attribute_lineage": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "confident": true
  },
  "overall": {
    "confident": true,
    "table_and_attribute_pass": true,
    "summary": "Table: 100.0% F1, Attribute: 100.0% F1"
  },
  "mapping_quality": {
    "total": 157,
    "by_type": {
      "INSERT_SELECT": 110,
      "NAME_MATCH_HEURISTIC": 47
    },
    "avg_confidence": 0.885,
    "high_confidence_count": 110
  }
}
```

### compliance_thresholds.json

```json
{
  "document_type": "Compliance Thresholds",
  "version": "4.2",
  "thresholds": {
    "confidence_threshold": 0.90,
    "mapping_type_priorities": {
      "INSERT_SELECT": 0.95,
      "UPDATE_SET": 0.90,
      "UPDATE_SET_INFERRED": 0.80,
      "NAME_MATCH_HEURISTIC": 0.70
    }
  },
  "validation_results": {...},
  "compliance_status": {
    "overall_compliant": true,
    "table_lineage_compliant": true,
    "attribute_lineage_compliant": true
  }
}
```

### compliance_summary.txt

Human-readable text format:
```
======================================================================
LINEAGE PIPELINE COMPLIANCE REPORT
======================================================================

Generated: 2025-11-22T...
Version: 4.2

----------------------------------------------------------------------
COMPLIANCE THRESHOLDS
----------------------------------------------------------------------

  Confidence Threshold:    90%
  Minimum Precision:       90%
  ...

----------------------------------------------------------------------
COMPLIANCE STATUS
----------------------------------------------------------------------

  Overall Compliant:           PASS
  Table Lineage Compliant:     PASS
  Attribute Lineage Compliant: PASS
```

## Interpreting Results

### PASS Status
Both table and attribute lineage meet the confidence threshold (default 90%):
- Precision >= 90%
- Recall >= 90%

### REVIEW REQUIRED Status
One or more metrics below threshold. Check:
1. `false_negatives`: Missing edges that should exist
2. `false_positives`: Extra edges that shouldn't exist
3. `mapping_quality`: Low-confidence mappings

## Pipeline Complete

After Validate Agent completes, the full pipeline is finished:

```
Parse -> Abstract -> Retrieve -> Generate -> Validate
  |         |           |           |           |
  v         v           v           v           v
parse_    abstract_   retrieve_  generate_   validate_
output    output      output     output      output
.json     .json       .json      .json       .json
```

All outputs are in:
- `output_artifacts/` - Final deliverables (YAML, Excel, CSV, validation report)
- `intermediate/` - Pipeline stage outputs
- `documentation/` - Compliance documentation
