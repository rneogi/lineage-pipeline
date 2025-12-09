# Parse Agent - PHASE 1 (v4.2)

## Overview

The Parse Agent is the first stage of the lineage extraction pipeline. It extracts and parses:

1. **DDL (Data Definition Language)** - Table schemas and column definitions
2. **Stored Procedures** - SQL code with deterministic attribute lineage extraction
3. **Parameter Lineage** - @param → table.column mappings (NEW in v4.2)
4. **Imperva Telemetry** - Runtime execution logs for usage patterns

## Usage

```bash
cd source_code
python parse_agent.py
```

## Input Files

| File | Location | Description |
|------|----------|-------------|
| `store-procedure-table.zip` | `input_data/` | ZIP containing DDL and stored procedure files |
| `imperva_small.xlsx` or `imperva_large.xlsx` | `input_data/` | Runtime telemetry logs (optional) |

### ZIP Contents Expected

```
store-procedure-table/
  iberia table extract.txt   # DDL file
  Iberia-PROC.txt            # Stored procedures file
```

## Output

| File | Location | Description |
|------|----------|-------------|
| `parse_output.json` | `intermediate/` | JSON containing all parsed data |

### Output Schema

```json
{
  "phase": "PARSE",
  "timestamp": "2025-11-22T...",
  "version": "4.2",
  "stats": {
    "tables": 2158,
    "columns": 2317,
    "procedures": 100,
    "attribute_mappings": 157,
    "parameter_mappings": 25,
    "high_confidence_mappings": 110,
    "imperva_logs": 138,
    "mapping_types": {
      "INSERT_SELECT": 110,
      "NAME_MATCH_HEURISTIC": 47
    }
  },
  "catalog": {
    "table_name": [
      {"name": "column_name", "type": "varchar(50)", "nullable": true}
    ]
  },
  "procedures": [
    {
      "name": "PROCEDURE_NAME",
      "all_tables": ["table1", "table2"],
      "source_tables": ["table1"],
      "target_tables": ["table2"],
      "attribute_lineage": [...],
      "parameter_lineage": [
        {
          "parameter": "@clm_no",
          "table": "tclmdets",
          "column": "clm_no",
          "via_procedure": "PROCEDURE_NAME",
          "pattern": "t.col = @p",
          "confidence": 0.9
        }
      ]
    }
  ],
  "imperva": {
    "logs": [...],
    "telemetry": {...}
  }
}
```

## Components

### DDLParser

Parses SQL DDL statements to extract table catalog:

- Extracts `CREATE TABLE` statements
- Parses column names, types, and nullability
- Builds normalized table-column mapping

### EnhancedSPParser (v4.2)

Parses stored procedures with deterministic attribute lineage:

| Mapping Type | Confidence | Description |
|--------------|------------|-------------|
| `INSERT_SELECT` | 0.95 | `INSERT INTO t (cols) SELECT cols FROM s` |
| `UPDATE_SET` | 0.90 | `UPDATE t SET col = s.col` |
| `UPDATE_SET_INFERRED` | 0.80 | `UPDATE t SET col = expr` with inferred source |
| `SELECT_INTO` | 0.95 | `SELECT cols INTO t FROM s` |
| `MERGE` | 0.95 | `MERGE INTO t USING s` |
| `NAME_MATCH_HEURISTIC` | 0.70 | Fallback for matching column names |

**NEW in v4.2:**
- `extract_parameter_lineage()` - Extracts @param → table.column mappings
- `_build_alias_map()` - Resolves table aliases (e.g., `FROM tclmdets cd` → `cd` = `tclmdets`)
- `_extract_param_declarations()` - Parses procedure header for @param names

### Parameter Lineage Patterns

The parser detects these patterns:

```sql
-- Pattern 1: t.col = @param
WHERE cd.clm_no = @clm_no

-- Pattern 2: @param = t.col
SET @total = cd.amount
```

### ImpervaLogParser

Parses runtime telemetry logs:

- Table access frequencies
- Procedure call frequencies
- Table-procedure co-occurrence patterns
- Telemetry-based confidence boosting

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_sp_chars` | 500,000 | Maximum characters to read from SP file |
| `max_procedures` | 100 | Maximum number of procedures to parse |

## Telemetry Confidence Boosting

When telemetry confirms lineage:
- **Strong confirmation** (both source + target + procedure in telemetry): +15% confidence
- **Partial confirmation** (some matches): +10% confidence

## Next Stage

After Parse Agent completes, run the **Abstract Agent** to build the YAML knowledge graph:

```bash
python abstract_agent.py
```
