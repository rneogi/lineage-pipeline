# Generate Agent - PHASE 4 (v4.2)

## Overview

The Generate Agent creates CSV and Excel reports for table and attribute lineage data. It produces formatted, sorted output files ready for analysis and sharing.

## Usage

```bash
cd source_code
python generate_agent.py
```

**Prerequisite:** Run `retrieve_agent.py` first to generate the input file.

## Input Files

| File | Location | Description |
|------|----------|-------------|
| `retrieve_output.json` | `intermediate/` | Output from Retrieve Agent (contains YAML graph) |

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `table_lineage.csv` | `output_artifacts/out_lineage/` | Table lineage in CSV format |
| `attribute_lineage.csv` | `output_artifacts/out_lineage/` | Attribute lineage in CSV format |
| `table_lineage.xlsx` | `output_artifacts/out_lineage/` | Formatted Excel with table lineage |
| `attribute_lineage.xlsx` | `output_artifacts/out_lineage/` | Formatted Excel with attribute lineage |
| `generate_output.json` | `intermediate/` | JSON summary for Validate Agent |

## Report Formats

### Table Lineage CSV

```csv
Source Table,Target Table,Via Procedure,Telemetry Score
taddress,tclmrec,AE_200_doc_NORWAY_A,0.04487179487179487
tpolprdr,tclmrec,AE_200_doc_NORWAY_A,0.019230769230769232
tclmdets,tlprcntl,AAA_BORDEREAU_DETAILS_LOAD,0.012820512820512822
```

**Columns:**
- `Source Table`: Table being read from
- `Target Table`: Table being written to
- `Via Procedure`: Stored procedure performing the operation
- `Telemetry Score`: Normalized score based on runtime access frequency

### Attribute Lineage CSV

```csv
Source Table,Source Column,Target Table,Target Column,Via Procedure,Confidence,Mapping Type,Telemetry Score
tclmdets,clm_office_cd,tlprdets,clm_office_cd,AAA_BORDEREAU_DETAILS_LOAD,0.95,INSERT_SELECT,0.019
```

**Columns:**
- `Source Table/Column`: Source of the data
- `Target Table/Column`: Destination of the data
- `Via Procedure`: Stored procedure performing the mapping
- `Confidence`: Mapping confidence (0.0 to 1.0)
- `Mapping Type`: How the mapping was detected
- `Telemetry Score`: Normalized runtime access score

### Mapping Types

| Type | Confidence | Description |
|------|------------|-------------|
| `INSERT_SELECT` | 0.95 | Explicit INSERT...SELECT with column mapping |
| `UPDATE_SET` | 0.90 | Explicit UPDATE SET col = table.col |
| `UPDATE_SET_INFERRED` | 0.80 | UPDATE with inferred source column |
| `SELECT_INTO` | 0.95 | SELECT cols INTO table FROM source |
| `MERGE` | 0.95 | MERGE INTO target USING source |
| `NAME_MATCH_HEURISTIC` | 0.70 | Fallback: matching column names |

## Excel Features

### Formatting
- Header row with blue background and white text
- Auto-sized column widths
- Cell borders for readability
- Sorted by Telemetry Score (table) or Confidence (attribute)

### Conditional Formatting (Attribute Lineage)
- Confidence column has color scale:
  - Red (low): < 70%
  - Yellow (medium): 70-90%
  - Green (high): > 90%

## Sorting

### Table Lineage
Sorted by `Telemetry Score` descending (most accessed first)

### Attribute Lineage
Sorted by:
1. `Confidence` descending (highest confidence first)
2. `Telemetry Score` descending (tie-breaker)

## Statistics Reported

After execution, the agent reports:
- Total table edges
- Total attribute edges
- Average confidence score
- Count of high-confidence mappings (>= 90%)
- Breakdown by mapping type

## Next Stage

After Generate Agent completes, run the **Validate Agent** for compliance validation:

```bash
python validate_agent.py
```
