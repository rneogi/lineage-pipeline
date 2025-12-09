# Quick Start Guide - demo.py v4.0

## 🚀 Getting Started in 5 Minutes

### 1. Install Dependencies

```bash
# Required
pip install pandas numpy scikit-learn openpyxl pyyaml --break-system-packages

# Optional but recommended (for better semantic search)
pip install sentence-transformers --break-system-packages

# Optional (for LLM chat interface)
pip install anthropic --break-system-packages
export ANTHROPIC_API_KEY='your-key-here'
```

### 2. Prepare Your Data

Place these files in your working directory:
```
your_project/
├── demo.py
├── store-procedure-table.zip    # DDL + Stored Procedures
├── imperva_small.xlsx           # Runtime logs (optional)
└── annotated_groundtruth.yaml   # Ground truth (optional)
```

### 3. Run the Pipeline

```bash
python demo.py
```

### 4. View Results

```bash
# Excel reports with deterministic mappings
open out_lineage/table_lineage.xlsx
open out_lineage/attribute_lineage.xlsx

# YAML knowledge graph
cat demo.enriched.yaml

# Validation report
cat validation_report.json
```

### 5. Use with Chat Interface

```bash
# Interactive chat
python chat_interface.py

# Example queries
You: Show me lineage for AAA_BORDEREAU_DETAILS_LOAD
You: What are the most frequently accessed tables?
You: Find high confidence attribute mappings
```

---

## 📊 Understanding the Output

### Excel Reports

**table_lineage.xlsx**:
- Source/Target tables
- Connecting procedures
- Telemetry scores (runtime importance)

**attribute_lineage.xlsx**:
- Column-to-column mappings
- **Confidence**: 0.70-0.95 (extraction method quality)
- **Mapping Type**: INSERT_SELECT, UPDATE_SET, etc.
- **Telemetry Score**: Runtime access frequency

### Confidence Levels Explained

| Confidence | Mapping Type | Meaning |
|-----------|-------------|---------|
| 0.95 | INSERT_SELECT | Explicit position-based mapping |
| 0.90 | UPDATE_SET | Qualified column in UPDATE statement |
| 0.80 | UPDATE_SET_INFERRED | Unqualified column (inferred from context) |
| 0.70 | NAME_MATCH_HEURISTIC | Same column name (fallback only) |

### Validation Report

```json
{
  "table_lineage": {
    "precision": 0.96,
    "recall": 0.93,
    "f1": 0.95,
    "confident": true
  },
  "attribute_lineage": {
    "precision": 0.92,
    "recall": 0.89,
    "f1": 0.90,
    "confident": true
  },
  "mapping_quality": {
    "total": 134,
    "avg_confidence": 0.918,
    "high_confidence_count": 120,
    "by_type": {
      "INSERT_SELECT": 78,
      "UPDATE_SET": 42,
      "UPDATE_SET_INFERRED": 10,
      "NAME_MATCH_HEURISTIC": 4
    }
  }
}
```

**What to look for**:
- ✅ F1 > 0.90 → Excellent quality
- ✅ High confidence count / total > 0.80 → Good coverage
- ⚠️ Many NAME_MATCH_HEURISTIC → Need better parser patterns

---

## 🔧 Configuration

### Process More Procedures

Edit `demo.py` line ~1091:
```python
for i, match in enumerate(proc_pattern.finditer(sp_text)):
    if i >= 10:  # Change to 50 or 100
        break
```

### Adjust Confidence Threshold

Edit `demo.py` in `Config` class:
```python
confidence_threshold: float = 0.90  # Change to 0.85 or 0.95
```

### Enable/Disable Telemetry Boosting

In chat queries:
```python
# Full telemetry
results = rag.query("query", boost_telemetry=True, telemetry_weight=0.3)

# No telemetry (pure semantic)
results = rag.query("query", boost_telemetry=False)
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'sentence_transformers'"
**Solution**: 
```bash
pip install sentence-transformers --break-system-packages
```
**Workaround**: Pipeline auto-falls back to TF-IDF

### Issue: Low precision (<70%)
**Cause**: Parser patterns don't match your SQL dialect
**Solution**: Add custom patterns in `extract_deterministic_attribute_lineage()`

### Issue: Low recall (<70%)
**Cause**: Not processing enough procedures
**Solution**: Increase procedure limit (see Configuration above)

### Issue: "No ground truth available"
**Effect**: Validation does self-check (100% metrics)
**Solution**: Create annotated_groundtruth.yaml with expected lineage

### Issue: All telemetry scores are 0.0
**Cause**: No Imperva logs or wrong file path
**Solution**: Check imperva_small.xlsx exists and has data

---

## 📈 Interpreting Results

### Good Results

```
📊 ATTRIBUTE LINEAGE
   Precision: 92.5%  ✅
   Recall:    88.7%  ✅
   F1 Score:  90.6%  ✅
   
   Mapping Quality:
      INSERT_SELECT: 78      (good)
      UPDATE_SET: 42         (good)
      UPDATE_SET_INFERRED: 10 (acceptable)
      NAME_MATCH_HEURISTIC: 4 (minimal fallback)
```

**Interpretation**: 
- High precision/recall → Parser extracts accurate mappings
- Most mappings are INSERT_SELECT or UPDATE_SET → High confidence
- Few heuristic fallbacks → Good SQL parsing

### Needs Improvement

```
📊 ATTRIBUTE LINEAGE
   Precision: 65.3%  ⚠️
   Recall:    58.2%  ⚠️
   F1 Score:  61.5%  ❌
   
   Mapping Quality:
      INSERT_SELECT: 12      (too few)
      UPDATE_SET: 8          (too few)
      UPDATE_SET_INFERRED: 5
      NAME_MATCH_HEURISTIC: 89 (too many!)
```

**Interpretation**:
- Low precision/recall → Parser misses patterns
- Many heuristic mappings → SQL patterns not recognized
- **Action**: Add more extraction patterns for your SQL dialect

---

## 🎯 Common Queries for Chat Interface

```bash
# Table discovery
"What tables are in the catalog?"
"Show me all tables"

# Procedure exploration
"What does AAA_BORDEREAU_DETAILS_LOAD do?"
"Which procedures write to tlprdets?"

# Lineage queries
"Show me data flow from tnumgen to tlprdets"
"What are the source tables for tlprdets?"

# Telemetry-based queries
"Show me the most frequently accessed tables"
"Which procedures are called most often?"
"What's the most important lineage?"

# Confidence filtering
"Show me high confidence attribute lineage"
"Find mappings with confidence > 90%"
"Which mappings are INSERT_SELECT type?"
```

---

## 📚 Next Steps

1. **Review Excel Reports**: Look for unexpected mappings
2. **Check Confidence Distribution**: Should be mostly >0.85
3. **Validate Against Known Lineage**: Compare with manual analysis
4. **Extend Parser Patterns**: Add SQL patterns specific to your system
5. **Process More Procedures**: Increase coverage for better recall

---

## 🆘 Getting Help

**Check these first:**
1. V4_UPGRADE_SUMMARY.md - Detailed technical explanation
2. demo.py inline comments - Implementation details
3. validation_report.json - Detailed metrics

**Common questions:**
- "Why is F1 low?" → See Troubleshooting section
- "How to add patterns?" → See V4_UPGRADE_SUMMARY.md Technical Deep Dive
- "What's a good F1 score?" → >90% with good ground truth, >80% without

---

**Version**: 4.0
**Last Updated**: November 22, 2025
