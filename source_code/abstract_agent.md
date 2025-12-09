# Abstract Agent - PHASE 2 (v4.2)

## Overview

The Abstract Agent builds a YAML Knowledge Graph from the parsed data, enriched with Imperva telemetry annotations. It transforms raw parsed data into a structured graph representation.

## Usage

```bash
cd source_code
python abstract_agent.py
```

**Prerequisite:** Run `parse_agent.py` first to generate the input file.

## Input Files

| File | Location | Description |
|------|----------|-------------|
| `parse_output.json` | `intermediate/` | Output from Parse Agent |

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `demo.enriched.yaml` | `output_artifacts/` | YAML Knowledge Graph with telemetry |
| `abstract_output.json` | `intermediate/` | JSON summary for next stage |

## Knowledge Graph Structure (v4.2)

### YAML Schema

```yaml
metadata:
  generated_at: "2025-11-22T..."
  version: "4.2"
  source: "Iberia Legacy System"
  features:
    - Deterministic attribute lineage
    - Parameter lineage extraction
    - Telemetry-enriched confidence scores
    - No cross-product mappings

catalog:
  table_name:
    - name: column_name
      type: varchar(50)
      nullable: true

procedures:
  - name: PROCEDURE_NAME
    tables: [table1, table2]
    source_tables: [table1]
    target_tables: [table2]
    attribute_mappings_count: 5
    parameter_mappings_count: 3

lineage:
  table_level:
    - source: table1
      target: table2
      via_procedure: PROCEDURE_NAME
      telemetry_score: 0.045

  attribute_level:
    - source_table: table1
      source_column: col1
      target_table: table2
      target_column: col1
      via_procedure: PROCEDURE_NAME
      confidence: 0.95
      mapping_type: INSERT_SELECT
      telemetry_score: 0.02

  parameter_level:  # NEW in v4.2
    - parameter: "@clm_no"
      table: tclmdets
      column: clm_no
      via_procedure: PROCEDURE_NAME
      pattern: "t.col = @p"
      confidence: 0.9
      telemetry_score: 0.03

telemetry:
  table_access_frequency:
    table_name: 100
  procedure_call_frequency:
    proc_name: 50
  total_log_entries: 138
```

## Telemetry Enrichment

The Abstract Agent enriches the knowledge graph with Imperva telemetry:

### Table-Level Lineage
- `telemetry_score` = average of (source_table_score + target_table_score + procedure_score)
- Scores are normalized (0.0 to 1.0) based on access frequency

### Attribute-Level Lineage
- `telemetry_score` = average of (source_table_score + target_table_score)
- `confidence` is boosted up to 5% based on telemetry score
- Higher access frequency = higher confidence boost

### Parameter-Level Lineage (NEW in v4.2)
- `telemetry_score` = table_importance_score from telemetry
- Links procedure parameters to table.column references

### Telemetry Score Calculation

```
table_score = table_access_count / max_table_access_count
procedure_score = procedure_call_count / max_procedure_call_count

telemetry_boost = telemetry_score * 0.05  # Up to 5% boost
confidence = min(0.99, base_confidence + telemetry_boost)
```

## Components

### TelemetryHelper

Computes normalized telemetry scores:
- `get_table_importance_score(table)` - Returns 0.0 to 1.0 based on access frequency
- `get_procedure_importance_score(proc)` - Returns 0.0 to 1.0 based on call frequency

### YAMLKnowledgeGraph

Builds the structured YAML representation:
- Processes catalog (tables and columns)
- Processes procedures with lineage
- Enriches edges with telemetry scores
- **NEW:** Adds parameter_level lineage edges
- Saves YAML output

## Output Statistics

After execution, the agent reports:
- Number of tables in catalog
- Number of procedures
- Number of table-level edges
- Number of attribute-level edges (deterministic, no cross-product)
- Number of parameter-level edges (NEW in v4.2)

## Next Stage

After Abstract Agent completes, run the **Retrieve Agent** to build the RAG library:

```bash
python retrieve_agent.py
```
