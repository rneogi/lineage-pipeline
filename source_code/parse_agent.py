"""
parse_agent.py - PHASE 1: PARSE Agent (v4.2)

Extracts DDL, Stored Procedures, and Imperva telemetry logs.
Outputs parsed data as JSON for the next pipeline stage.

Features (v4.2):
- Parse Sybase DDL into table/column catalog
- Parse Stored Procedures for:
    • Table usage (source/target tables)
    • Deterministic attribute lineage (INSERT SELECT, UPDATE SET, etc.)
    • Parameter → column lineage (@param = t.col patterns)
- Parse Imperva logs into telemetry (table/procedure frequencies)
- Table alias resolution for accurate mappings

Usage:
    python parse_agent.py

Outputs:
    intermediate/parse_output.json - Parsed catalog, procedures, and telemetry
"""

import os
import re
import json
import zipfile
import collections
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Input paths
    zip_path: Path = field(init=False)
    imperva_path: Path = field(init=False)

    # Output paths
    intermediate_dir: Path = field(init=False)
    parse_output: Path = field(init=False)

    # Processing parameters - INCREASED for better coverage
    max_sp_chars: int = 0  # 0 = no limit, read entire file
    max_procedures: int = 0   # 0 = no limit, parse ALL procedures for complete coverage

    def __post_init__(self):
        # Input files from input_data directory
        input_dir = self.base_dir / "input_data"
        demo_dir = input_dir / "demo"

        # New structure: files extracted from datax.zip into demo/
        # DDL file
        if (demo_dir / "iberia table extract.txt").exists():
            self.ddl_path = demo_dir / "iberia table extract.txt"
        else:
            self.ddl_path = None

        # Stored Procedures file
        if (demo_dir / "SPAIN - GOALD - Stored Procedures.txt").exists():
            self.sp_path = demo_dir / "SPAIN - GOALD - Stored Procedures.txt"
        else:
            self.sp_path = None

        # Imperva logs - now CSV format
        imperva_csv = list(demo_dir.glob("Imperva*.csv")) if demo_dir.exists() else []
        if imperva_csv:
            self.imperva_path = imperva_csv[0]
        elif (input_dir / "imperva_small.xlsx").exists():
            self.imperva_path = input_dir / "imperva_small.xlsx"
        else:
            self.imperva_path = None

        # Legacy zip support (fallback)
        if (input_dir / "store-procedure-table.zip").exists():
            self.zip_path = input_dir / "store-procedure-table.zip"
        elif (input_dir / "datax.zip").exists():
            self.zip_path = input_dir / "datax.zip"
        else:
            self.zip_path = None

        # Intermediate directory for outputs
        self.intermediate_dir = self.base_dir / "intermediate"
        self.parse_output = self.intermediate_dir / "parse_output.json"

        # Create directories
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DDL PARSER
# ============================================================================

class DDLParser:
    """Parses Sybase/SQL Server DDL to build table catalog"""

    def __init__(self):
        self.catalog: Dict[str, List[Dict[str, Any]]] = {}
        self.stats = {
            'tables': 0,
            'columns': 0,
            'foreign_keys': 0
        }

    def parse(self, ddl_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse DDL and return table catalog"""
        create_pat = re.compile(
            r"CREATE\s+TABLE\s+(?:dbo\.)?([A-Za-z0-9_\.\[\]]+)\s*\((.*?)\)(?:\s+LOCK\s+\w+)?(?:\s*GO)?",
            re.I | re.S
        )

        col_pat = re.compile(
            r"^\s*\[?([A-Za-z0-9_]+)\]?\s+([A-Za-z0-9\(\),]+)(?:\s+(NULL|NOT\s+NULL))?",
            re.I
        )

        for m in create_pat.finditer(ddl_text):
            raw_table = m.group(1).replace("[", "").replace("]", "")
            table_name = raw_table.lower()

            cols_block = m.group(2)
            cols = []

            for line in cols_block.splitlines():
                line = line.strip().rstrip(",")
                if not line or line.upper().startswith((
                    "CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "INDEX"
                )):
                    continue

                cm = col_pat.match(line)
                if not cm:
                    continue

                cname, ctype = cm.group(1), cm.group(2)
                nullable = cm.group(3)

                cols.append({
                    "name": cname.lower(),
                    "type": ctype.lower(),
                    "nullable": nullable != "NOT NULL" if nullable else True
                })

            if cols:
                self.catalog[table_name] = cols
                self.stats['tables'] += 1
                self.stats['columns'] += len(cols)

        return self.catalog


# ============================================================================
# ENHANCED SP PARSER
# ============================================================================

class EnhancedSPParser:
    """
    Enhanced Stored Procedure Parser with DETERMINISTIC column mappings.

    v4.2 Enhancements:
    - MERGE statement support (95% confidence)
    - SELECT INTO support (95% confidence)
    - CTE (WITH...AS) support (90% confidence)
    - Table alias resolution via _build_alias_map()
    - Parameter → column lineage extraction (@param = t.col patterns)
    - Telemetry-based confidence boosting
    """

    SQL_KEYWORDS = {
        'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE',
        'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL', 'CROSS',
        'ON', 'USING', 'AS', 'IS', 'NULL', 'BETWEEN', 'CASE', 'WHEN',
        'THEN', 'ELSE', 'END', 'IF', 'WHILE', 'BEGIN', 'RETURN',
        'GO', 'EXEC', 'EXECUTE', 'DECLARE', 'SET', 'PRINT', 'SELECT',
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'TABLE',
        'INTO', 'VALUES', 'TOP', 'DISTINCT', 'ORDER', 'BY', 'GROUP',
        'HAVING', 'UNION', 'ALL', 'DESC', 'ASC', 'MERGE', 'WHEN', 'MATCHED',
        'WITH', 'CTE', 'RECURSIVE', 'OUTPUT', 'INSERTED', 'DELETED'
    }

    def __init__(self, catalog: Dict[str, List[Dict]], telemetry: Dict = None):
        self.catalog = catalog
        self.catalog_lower = {k.lower(): k for k in catalog.keys()}
        self.column_map = self._build_column_map()
        self.table_columns = self._build_table_column_map()
        self.telemetry = telemetry or {}
        self.alias_map = {}  # Track table aliases per procedure

    def _build_column_map(self) -> Dict[str, Set[str]]:
        col_map = collections.defaultdict(set)
        for table, cols in self.catalog.items():
            for col in cols:
                col_map[col['name']].add(table)
        return col_map

    def _build_table_column_map(self) -> Dict[str, Set[str]]:
        return {
            table: {col['name'] for col in cols}
            for table, cols in self.catalog.items()
        }

    def is_valid_table_name(self, token: str) -> bool:
        if not token or len(token) < 3:
            return False
        if token.upper() in self.SQL_KEYWORDS:
            return False
        if token[0].isdigit():
            return False
        return token.lower() in self.catalog_lower

    def is_valid_column_name(self, token: str, table: str = None) -> bool:
        token_lower = token.lower()
        if token.upper() in self.SQL_KEYWORDS:
            return False
        if token_lower.isdigit():
            return False
        if table:
            return token_lower in self.table_columns.get(table, set())
        else:
            return token_lower in self.column_map

    def normalize_table_name(self, token: str) -> str:
        token = re.sub(r'^\w+\.', '', token)
        # Check alias map first
        if token.lower() in self.alias_map:
            return self.alias_map[token.lower()]
        return self.catalog_lower.get(token.lower(), token)

    def extract_aliases(self, sql: str) -> Dict[str, str]:
        """Extract table aliases from SQL (e.g., 'FROM tclmdets t1' -> {'t1': 'tclmdets'})"""
        self.alias_map = {}

        # Pattern: FROM/JOIN table alias or FROM/JOIN table AS alias
        alias_patterns = [
            # FROM table alias (without AS)
            r'\b(?:FROM|JOIN)\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\b(?!\s*\.)',
            # FROM table AS alias
            r'\b(?:FROM|JOIN)\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+AS\s+([A-Za-z][A-Za-z0-9_]*)',
            # UPDATE table alias SET
            r'\bUPDATE\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+([A-Za-z][A-Za-z0-9_]*)\s+SET',
        ]

        for pattern in alias_patterns:
            for m in re.finditer(pattern, sql, re.I):
                table_name = m.group(1).lower()
                alias = m.group(2).lower()

                # Skip if alias is a SQL keyword or same as table
                if alias.upper() in self.SQL_KEYWORDS:
                    continue
                if alias == table_name:
                    continue

                # Verify table exists in catalog
                if table_name in self.catalog_lower:
                    self.alias_map[alias] = self.catalog_lower[table_name]

        return self.alias_map

    def extract_tables(self, sql: str) -> List[Tuple[str, str]]:
        # First extract aliases
        self.extract_aliases(sql)
        results = []

        from_pat = re.compile(r'\b(?:FROM|JOIN)\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)', re.I)
        for m in from_pat.finditer(sql):
            raw = m.group(1)
            if self.is_valid_table_name(raw):
                results.append((self.normalize_table_name(raw), "SELECT"))

        insert_pat = re.compile(r'\bINSERT\s+INTO\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)', re.I)
        for m in insert_pat.finditer(sql):
            raw = m.group(1)
            if self.is_valid_table_name(raw):
                results.append((self.normalize_table_name(raw), "INSERT"))

        update_pat = re.compile(r'\bUPDATE\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+SET', re.I)
        for m in update_pat.finditer(sql):
            raw = m.group(1)
            if self.is_valid_table_name(raw):
                results.append((self.normalize_table_name(raw), "UPDATE"))

        delete_pat = re.compile(r'\bDELETE\s+FROM\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)', re.I)
        for m in delete_pat.finditer(sql):
            raw = m.group(1)
            if self.is_valid_table_name(raw):
                results.append((self.normalize_table_name(raw), "DELETE"))

        return results

    def extract_deterministic_attribute_lineage(
        self, sql: str, procedure_name: str
    ) -> List[Dict[str, Any]]:
        mappings = []
        tables_ops = self.extract_tables(sql)
        source_tables = {t for t, op in tables_ops if op == "SELECT"}
        target_tables = {t for t, op in tables_ops if op in ("INSERT", "UPDATE", "DELETE")}

        # PATTERN 1: INSERT INTO table (cols) SELECT cols FROM source
        insert_select_pat = re.compile(
            r'INSERT\s+INTO\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s*\((.*?)\)\s*SELECT\s+(.*?)\s+FROM',
            re.I | re.S
        )

        for m in insert_select_pat.finditer(sql):
            target_table = self.normalize_table_name(m.group(1))
            target_cols_str = m.group(2)
            select_cols_str = m.group(3)

            target_cols = [c.strip().lower() for c in target_cols_str.split(',')]
            target_cols = [re.sub(r'^\[?(.*?)\]?$', r'\1', c) for c in target_cols]

            select_items = [c.strip() for c in select_cols_str.split(',')]
            source_cols = []

            for item in select_items:
                item = re.sub(r'\s+AS\s+\w+', '', item, flags=re.I)
                qualified_match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)', item)
                if qualified_match:
                    col = qualified_match.group(2).lower()
                    source_cols.append(col)
                else:
                    col = re.sub(r'[^\w]', '', item).lower()
                    if self.is_valid_column_name(col):
                        source_cols.append(col)
                    else:
                        source_cols.append(None)

            for i, (tgt_col, src_col) in enumerate(zip(target_cols, source_cols)):
                if src_col and self.is_valid_column_name(src_col):
                    for src_table in source_tables:
                        if src_col in self.table_columns.get(src_table, set()):
                            if tgt_col in self.table_columns.get(target_table, set()):
                                mappings.append({
                                    'source_table': src_table,
                                    'source_column': src_col,
                                    'target_table': target_table,
                                    'target_column': tgt_col,
                                    'via_procedure': procedure_name,
                                    'confidence': 0.95,
                                    'mapping_type': 'INSERT_SELECT'
                                })

        # PATTERN 2: UPDATE table SET col = expr
        update_set_pat = re.compile(
            r'UPDATE\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+SET\s+(.*?)(?:WHERE|FROM|$)',
            re.I | re.S
        )

        for m in update_set_pat.finditer(sql):
            target_table = self.normalize_table_name(m.group(1))
            set_clause = m.group(2)
            assignments = [a.strip() for a in set_clause.split(',')]

            for assign in assignments:
                if '=' not in assign:
                    continue
                parts = assign.split('=', 1)
                if len(parts) != 2:
                    continue

                target_col = parts[0].strip().lower()
                target_col = re.sub(r'^\[?(.*?)\]?$', r'\1', target_col)

                if target_col not in self.table_columns.get(target_table, set()):
                    continue

                expr = parts[1].strip()
                qualified_matches = re.finditer(
                    r'([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)',
                    expr
                )

                for qm in qualified_matches:
                    src_table_or_alias = qm.group(1).lower()
                    src_col = qm.group(2).lower()

                    if src_table_or_alias in self.catalog_lower:
                        src_table = self.catalog_lower[src_table_or_alias]
                        if src_col in self.table_columns.get(src_table, set()):
                            mappings.append({
                                'source_table': src_table,
                                'source_column': src_col,
                                'target_table': target_table,
                                'target_column': target_col,
                                'via_procedure': procedure_name,
                                'confidence': 0.90,
                                'mapping_type': 'UPDATE_SET'
                            })

                unqualified_tokens = re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\b', expr)
                for token in unqualified_tokens:
                    token_lower = token.lower()
                    if token.upper() in self.SQL_KEYWORDS:
                        continue
                    if token_lower in self.column_map:
                        for src_table in source_tables:
                            if token_lower in self.table_columns.get(src_table, set()):
                                mappings.append({
                                    'source_table': src_table,
                                    'source_column': token_lower,
                                    'target_table': target_table,
                                    'target_column': target_col,
                                    'via_procedure': procedure_name,
                                    'confidence': 0.80,
                                    'mapping_type': 'UPDATE_SET_INFERRED'
                                })

        # PATTERN 3: SELECT INTO (creates new table or inserts)
        select_into_pat = re.compile(
            r'SELECT\s+(.*?)\s+INTO\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+FROM',
            re.I | re.S
        )
        for m in select_into_pat.finditer(sql):
            select_cols_str = m.group(1)
            target_table = self.normalize_table_name(m.group(2))

            if target_table not in self.catalog_lower.values():
                continue

            select_items = [c.strip() for c in select_cols_str.split(',')]
            for item in select_items:
                item = re.sub(r'\s+AS\s+\w+', '', item, flags=re.I)
                qualified_match = re.search(r'([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z][A-Za-z0-9_]*)', item)
                if qualified_match:
                    src_ref = qualified_match.group(1).lower()
                    src_col = qualified_match.group(2).lower()
                    src_table = self.normalize_table_name(src_ref)

                    if src_table in self.table_columns and src_col in self.table_columns.get(src_table, set()):
                        if src_col in self.table_columns.get(target_table, set()):
                            mappings.append({
                                'source_table': src_table,
                                'source_column': src_col,
                                'target_table': target_table,
                                'target_column': src_col,
                                'via_procedure': procedure_name,
                                'confidence': 0.95,
                                'mapping_type': 'SELECT_INTO'
                            })

        # PATTERN 4: MERGE statement
        merge_pat = re.compile(
            r'MERGE\s+(?:INTO\s+)?(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)\s+.*?USING\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)',
            re.I | re.S
        )
        for m in merge_pat.finditer(sql):
            target_table = self.normalize_table_name(m.group(1))
            source_table = self.normalize_table_name(m.group(2))

            if target_table in self.table_columns and source_table in self.table_columns:
                src_cols = self.table_columns.get(source_table, set())
                tgt_cols = self.table_columns.get(target_table, set())
                common_cols = src_cols & tgt_cols
                for col in common_cols:
                    mappings.append({
                        'source_table': source_table,
                        'source_column': col,
                        'target_table': target_table,
                        'target_column': col,
                        'via_procedure': procedure_name,
                        'confidence': 0.95,
                        'mapping_type': 'MERGE'
                    })

        # PATTERN 5: CTE (WITH...AS) - track columns through CTEs
        cte_pat = re.compile(
            r'WITH\s+([A-Za-z][A-Za-z0-9_]*)\s+AS\s*\(\s*SELECT\s+(.*?)\s+FROM\s+(?:dbo\.)?([A-Za-z][A-Za-z0-9_]*)',
            re.I | re.S
        )
        for m in cte_pat.finditer(sql):
            cte_name = m.group(1).lower()
            select_cols_str = m.group(2)
            source_table = self.normalize_table_name(m.group(3))

            if source_table in self.table_columns:
                # Track CTE as virtual table for downstream references
                source_tables.add(source_table)

        # PATTERN 6: Name match heuristic fallback (only if no high-confidence mappings found)
        high_confidence_found = any(m['confidence'] >= 0.90 for m in mappings)
        if not high_confidence_found and source_tables and target_tables:
            for src_table in source_tables:
                for tgt_table in target_tables:
                    src_cols = self.table_columns.get(src_table, set())
                    tgt_cols = self.table_columns.get(tgt_table, set())
                    common_cols = src_cols & tgt_cols
                    for col in common_cols:
                        # Check if this mapping already exists
                        exists = any(
                            m['source_table'] == src_table and
                            m['source_column'] == col and
                            m['target_table'] == tgt_table and
                            m['target_column'] == col
                            for m in mappings
                        )
                        if not exists:
                            mappings.append({
                                'source_table': src_table,
                                'source_column': col,
                                'target_table': tgt_table,
                                'target_column': col,
                                'via_procedure': procedure_name,
                                'confidence': 0.70,
                                'mapping_type': 'NAME_MATCH_HEURISTIC'
                            })

        # Apply telemetry boost if available
        mappings = self._apply_telemetry_boost(mappings, procedure_name)

        return mappings

    def _extract_param_declarations(self, sp_text: str) -> List[str]:
        """
        Extract parameter names from stored procedure header.
        Looks for '@name' tokens between CREATE PROCEDURE and 'AS'.
        """
        header_match = re.search(
            r"CREATE\s+PROCEDURE\s+(?:dbo\.)?[A-Za-z0-9_]+\s*(.*?)\bAS\b",
            sp_text, re.I | re.S
        )
        if not header_match:
            return []
        header = header_match.group(1)
        return sorted({p.lower() for p in re.findall(r"@([A-Za-z0-9_]+)", header)})

    def _build_alias_map(self, sp_text: str) -> Dict[str, str]:
        """
        Build alias -> base table mapping across the procedure.
        Handles patterns like: FROM tclmdets cd, JOIN tnumgen tn, etc.
        """
        alias_map: Dict[str, str] = {}
        pat = re.compile(
            r"\b(?:FROM|JOIN)\s+(?:dbo\.)?([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)",
            re.I
        )
        for m in pat.finditer(sp_text):
            raw_table = m.group(1)
            alias = m.group(2).lower()
            # Skip if alias is a SQL keyword
            if alias.upper() in self.SQL_KEYWORDS:
                continue
            tnorm = self.normalize_table_name(raw_table)
            if tnorm in self.catalog_lower.values():
                alias_map[alias] = tnorm
        return alias_map

    def extract_parameter_lineage(self, sp_text: str, proc_name: str) -> List[Dict[str, Any]]:
        """
        Extract parameter → column lineage from patterns like:
        - WHERE t.col = @param
        - @param = t.col

        Resolves table aliases via _build_alias_map().
        """
        params = self._extract_param_declarations(sp_text)
        if not params:
            return []

        alias_map = self._build_alias_map(sp_text)
        results: List[Dict[str, Any]] = []

        for p in params:
            p_tok = f"@{p}"

            # Pattern 1: t.col = @param
            pat1 = re.compile(
                rf"([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*{re.escape(p_tok)}\b",
                re.I
            )
            # Pattern 2: @param = t.col
            pat2 = re.compile(
                rf"{re.escape(p_tok)}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)",
                re.I
            )

            for m in pat1.finditer(sp_text):
                alias, col = m.group(1).lower(), m.group(2).lower()
                tbl = alias_map.get(alias)
                # If not in alias map, check if it's a direct table name
                if not tbl and alias in self.catalog_lower:
                    tbl = self.catalog_lower[alias]
                results.append({
                    "parameter": p_tok.lower(),
                    "table": tbl,
                    "column": col,
                    "via_procedure": proc_name,
                    "pattern": "t.col = @p",
                    "confidence": 0.9 if tbl else 0.6
                })

            for m in pat2.finditer(sp_text):
                alias, col = m.group(1).lower(), m.group(2).lower()
                tbl = alias_map.get(alias)
                if not tbl and alias in self.catalog_lower:
                    tbl = self.catalog_lower[alias]
                results.append({
                    "parameter": p_tok.lower(),
                    "table": tbl,
                    "column": col,
                    "via_procedure": proc_name,
                    "pattern": "@p = t.col",
                    "confidence": 0.9 if tbl else 0.6
                })

        return results

    def _apply_telemetry_boost(self, mappings: List[Dict], procedure_name: str) -> List[Dict]:
        """Boost confidence by 10-15% if telemetry confirms the lineage"""
        if not self.telemetry:
            return mappings

        table_access = self.telemetry.get('table_access_freq', {})
        cooccurrence = self.telemetry.get('table_proc_cooc', {})
        proc_freq = self.telemetry.get('procedure_call_freq', {})

        proc_lower = procedure_name.lower()
        proc_is_active = proc_lower in proc_freq

        for mapping in mappings:
            src_table = mapping['source_table'].lower()
            tgt_table = mapping['target_table'].lower()

            # Check if telemetry confirms this lineage (using tuple keys)
            src_confirmed = (src_table, proc_lower) in cooccurrence or src_table in table_access
            tgt_confirmed = (tgt_table, proc_lower) in cooccurrence or tgt_table in table_access

            if src_confirmed and tgt_confirmed and proc_is_active:
                # Strong telemetry confirmation - boost by 15%
                mapping['confidence'] = min(1.0, mapping['confidence'] + 0.15)
                mapping['telemetry_confirmed'] = True
            elif src_confirmed or tgt_confirmed:
                # Partial confirmation - boost by 10%
                mapping['confidence'] = min(1.0, mapping['confidence'] + 0.10)
                mapping['telemetry_confirmed'] = 'partial'

        return mappings

    def parse_procedure(self, sp_text: str) -> Dict[str, Any]:
        name_match = re.search(
            r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(?:dbo\.)?([A-Za-z0-9_]+)',
            sp_text,
            re.I
        )
        proc_name = name_match.group(1) if name_match else "unknown_procedure"

        tables = self.extract_tables(sp_text)
        unique_tables = list({t[0] for t in tables})

        source_tables = {t[0] for t in tables if t[1] == "SELECT"}
        target_tables = {t[0] for t in tables if t[1] in ("INSERT", "UPDATE", "DELETE")}

        attribute_lineage = self.extract_deterministic_attribute_lineage(sp_text, proc_name)
        parameter_lineage = self.extract_parameter_lineage(sp_text, proc_name)

        return {
            'name': proc_name,
            'all_tables': unique_tables,
            'source_tables': list(source_tables),
            'target_tables': list(target_tables),
            'attribute_lineage': attribute_lineage,
            'parameter_lineage': parameter_lineage,
            'raw_tables': tables
        }


# ============================================================================
# IMPERVA LOG PARSER
# ============================================================================

class ImpervaLogParser:
    """Parse Imperva runtime logs for actual execution patterns (CSV or Excel)."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.logs: List[Dict[str, Any]] = []
        self.telemetry = {
            'table_access_freq': {},
            'procedure_call_freq': {},
            'table_proc_cooc': collections.Counter(),
            'sql_patterns': []
        }

    def parse(self) -> List[Dict[str, Any]]:
        if self.log_path is None or not self.log_path.exists():
            print(f"Warning: Imperva log not found: {self.log_path}")
            return []

        try:
            path_str = str(self.log_path).lower()

            if path_str.endswith('.csv'):
                # CSV format (new)
                df = pd.read_csv(self.log_path, encoding='utf-8', on_bad_lines='skip')
            elif path_str.endswith('.zip'):
                import io
                with zipfile.ZipFile(self.log_path, 'r') as z:
                    xlsx_files = [f for f in z.namelist() if f.endswith('.xlsx')]
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if xlsx_files:
                        with z.open(xlsx_files[0]) as f:
                            df = pd.read_excel(io.BytesIO(f.read()))
                    elif csv_files:
                        with z.open(csv_files[0]) as f:
                            df = pd.read_csv(io.BytesIO(f.read()), encoding='utf-8', on_bad_lines='skip')
                    else:
                        print(f"Warning: No .xlsx or .csv file found in {self.log_path}")
                        return []
            else:
                # Excel format (legacy)
                df = pd.read_excel(self.log_path)

            # Normalize column names (handle different naming conventions)
            col_map = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'timestamp' in col_lower or 'time' in col_lower:
                    col_map[col] = 'timestamp'
                elif 'user' in col_lower and 'db' in col_lower:
                    col_map[col] = 'db_user'
                elif 'object' in col_lower and 'type' in col_lower:
                    col_map[col] = 'object_type'
                elif 'operation' in col_lower:
                    col_map[col] = 'operation_type'
                elif 'source' in col_lower and 'program' in col_lower:
                    col_map[col] = 'source_program'
                elif 'objects' in col_lower and 'verbs' in col_lower:
                    col_map[col] = 'objects_verbs'
                elif 'sql' in col_lower or 'query' in col_lower:
                    col_map[col] = 'original_sql'
                elif col_lower == 'count':
                    col_map[col] = 'count'

            for _, row in df.iterrows():
                log_entry = {
                    'timestamp': str(row.get(next((k for k, v in col_map.items() if v == 'timestamp'), 'First Timestamp'), '')),
                    'db_user': str(row.get(next((k for k, v in col_map.items() if v == 'db_user'), '_id_DB User Name'), '')),
                    'object_type': str(row.get(next((k for k, v in col_map.items() if v == 'object_type'), '_id_Object Type'), '')),
                    'operation_type': str(row.get(next((k for k, v in col_map.items() if v == 'operation_type'), '_id_Operation Type'), '')),
                    'source_program': str(row.get(next((k for k, v in col_map.items() if v == 'source_program'), '_id_Source Program'), '')),
                    'objects_verbs': str(row.get(next((k for k, v in col_map.items() if v == 'objects_verbs'), '_id_Objects and Verbs'), '')),
                    'original_sql': str(row.get(next((k for k, v in col_map.items() if v == 'original_sql'), '_id_Original SQL'), '')),
                    'count': 1
                }
                # Try to get count
                count_col = next((k for k, v in col_map.items() if v == 'count'), 'count')
                if count_col in row and pd.notna(row.get(count_col)):
                    try:
                        log_entry['count'] = int(row.get(count_col))
                    except (ValueError, TypeError):
                        log_entry['count'] = 1

                self.logs.append(log_entry)

            print(f"Parsed {len(self.logs)} Imperva log entries")
            self._build_telemetry()
            return self.logs

        except Exception as e:
            print(f"Warning: Error parsing Imperva log: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _build_telemetry(self):
        for log in self.logs:
            count = log.get('count', 1)
            sql = str(log.get('original_sql', '')).lower()

            if not sql or sql == 'nan':
                continue

            table_pat = re.compile(r'\b(?:from|join|into|update)\s+(?:dbo\.)?([a-z][a-z0-9_]+)', re.I)
            tables = {m.group(1).lower() for m in table_pat.finditer(sql)}

            for table in tables:
                self.telemetry['table_access_freq'][table] = \
                    self.telemetry['table_access_freq'].get(table, 0) + count

            proc_pat = re.compile(r'(?:exec|execute)\s+([a-z][a-z0-9_]+)', re.I)
            procs = {m.group(1).lower() for m in proc_pat.finditer(sql)}

            for proc in procs:
                self.telemetry['procedure_call_freq'][proc] = \
                    self.telemetry['procedure_call_freq'].get(proc, 0) + count

                for table in tables:
                    self.telemetry['table_proc_cooc'][(table, proc)] += count

            if len(self.telemetry['sql_patterns']) < 100:
                self.telemetry['sql_patterns'].append({
                    'sql': sql[:500],
                    'tables': list(tables),
                    'procedures': list(procs),
                    'count': count
                })

        print(f"Built telemetry: {len(self.telemetry['table_access_freq'])} tables, "
              f"{len(self.telemetry['procedure_call_freq'])} procedures tracked")

    def to_dict(self) -> Dict[str, Any]:
        """Export telemetry as dictionary for JSON serialization"""
        # Convert tuple keys in table_proc_cooc to strings for JSON compatibility
        cooc_serializable = {
            f"{k[0]}|{k[1]}": v for k, v in self.telemetry['table_proc_cooc'].items()
        }
        telemetry_copy = dict(self.telemetry)
        telemetry_copy['table_proc_cooc'] = cooc_serializable

        return {
            'logs': self.logs,
            'telemetry': telemetry_copy
        }


# ============================================================================
# MAIN PARSE AGENT
# ============================================================================

def main():
    """Execute Parse Agent"""
    print("=" * 70)
    print("PHASE 1: PARSE AGENT")
    print("=" * 70)
    print("Extracting DDL, Stored Procedures, and Imperva telemetry")
    print("-" * 70)

    config = Config()

    # Determine DDL and SP paths
    ddl_path = None
    sp_path = None

    # Check if new structure exists (files already extracted)
    if config.ddl_path and config.ddl_path.exists():
        ddl_path = config.ddl_path
        sp_path = config.sp_path
        print(f"\nUsing pre-extracted files from: {config.ddl_path.parent}")
    elif config.zip_path and config.zip_path.exists():
        # Fallback: Extract from ZIP
        sp_demo_root = config.intermediate_dir / "sp_demo"
        sp_demo_root.mkdir(exist_ok=True)

        print(f"\nExtracting {config.zip_path}...")
        with zipfile.ZipFile(config.zip_path, "r") as z:
            z.extractall(sp_demo_root)

        # Try different possible paths from various zip structures
        possible_ddl = [
            sp_demo_root / "store-procedure-table" / "iberia table extract.txt",
            sp_demo_root / "demo" / "iberia table extract.txt",
            sp_demo_root / "iberia table extract.txt",
        ]
        possible_sp = [
            sp_demo_root / "store-procedure-table" / "Iberia-PROC.txt",
            sp_demo_root / "demo" / "SPAIN - GOALD - Stored Procedures.txt",
            sp_demo_root / "SPAIN - GOALD - Stored Procedures.txt",
        ]

        for p in possible_ddl:
            if p.exists():
                ddl_path = p
                break
        for p in possible_sp:
            if p.exists():
                sp_path = p
                break
    else:
        print("ERROR: No input files found. Please extract datax.zip to input_data/demo/")
        return

    if not ddl_path or not ddl_path.exists():
        print(f"ERROR: DDL file not found")
        return
    if not sp_path or not sp_path.exists():
        print(f"ERROR: Stored Procedures file not found")
        return

    # Parse DDL
    print(f"\nParsing DDL from: {ddl_path.name}...")
    ddl_text = ddl_path.read_text(encoding="utf-8", errors="ignore")
    ddl_parser = DDLParser()
    catalog = ddl_parser.parse(ddl_text)
    print(f"Parsed {ddl_parser.stats['tables']} tables, {ddl_parser.stats['columns']} columns")

    # Parse Imperva logs
    print("\nParsing Imperva telemetry logs...")
    imperva_parser = ImpervaLogParser(config.imperva_path)
    imperva_parser.parse()

    # Parse Stored Procedures (with telemetry for confidence boosting)
    print(f"\nParsing stored procedures from: {sp_path.name}...")
    print("(with deterministic mappings + telemetry boost)...")
    sp_text = sp_path.read_text(encoding="utf-8", errors="ignore")
    if config.max_sp_chars > 0:
        sp_text = sp_text[:config.max_sp_chars]
    print(f"  File size: {len(sp_text):,} characters")

    # Pass telemetry to SP parser for confidence boosting
    telemetry = imperva_parser.telemetry if imperva_parser.logs else {}
    sp_parser = EnhancedSPParser(catalog, telemetry=telemetry)

    # Get telemetry procedure names for prioritization
    telemetry_procs = set(telemetry.get('procedure_call_freq', {}).keys())
    print(f"  Telemetry procedures to prioritize: {len(telemetry_procs)}")

    # First pass: collect all procedure blocks with their names
    proc_pattern = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE.*?(?=CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE|\Z)',
        re.I | re.S
    )
    name_pattern = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(?:dbo\.)?([A-Za-z0-9_]+)',
        re.I
    )

    all_proc_blocks = []
    for match in proc_pattern.finditer(sp_text):
        sp_block = match.group(0)
        name_match = name_pattern.search(sp_block)
        proc_name = name_match.group(1).lower() if name_match else "unknown"
        all_proc_blocks.append((proc_name, sp_block))

    print(f"  Total procedures found: {len(all_proc_blocks)}")

    # Prioritize: telemetry procedures first, then others
    telemetry_blocks = [(n, b) for n, b in all_proc_blocks if n in telemetry_procs]
    other_blocks = [(n, b) for n, b in all_proc_blocks if n not in telemetry_procs]

    # Sort telemetry blocks by call frequency (most called first)
    proc_freq = telemetry.get('procedure_call_freq', {})
    telemetry_blocks.sort(key=lambda x: proc_freq.get(x[0], 0), reverse=True)

    # Combine: all telemetry procs + remaining procs up to limit (0 = no limit)
    if config.max_procedures <= 0:
        # No limit - process ALL procedures
        prioritized_blocks = telemetry_blocks + other_blocks
    else:
        prioritized_blocks = telemetry_blocks + other_blocks[:max(0, config.max_procedures - len(telemetry_blocks))]
    print(f"  Processing {len(telemetry_blocks)} telemetry + {len(prioritized_blocks) - len(telemetry_blocks)} other = {len(prioritized_blocks)} procedures")

    procedures = []
    print("\nExtracting procedures with deterministic attribute + parameter lineage...")
    high_conf_count = 0
    total_param_mappings = 0
    for i, (proc_name, sp_block) in enumerate(prioritized_blocks):
        proc_data = sp_parser.parse_procedure(sp_block)
        procedures.append(proc_data)

        # Count high-confidence mappings
        proc_high_conf = sum(1 for m in proc_data.get('attribute_lineage', []) if m.get('confidence', 0) >= 0.90)
        high_conf_count += proc_high_conf

        mappings = proc_data.get('attribute_lineage', [])
        param_maps = proc_data.get('parameter_lineage', [])
        total_param_mappings += len(param_maps)
        telemetry_boosted = sum(1 for m in mappings if m.get('telemetry_confirmed'))
        print(f"  {i+1}. {proc_data['name']}: {len(mappings)} attr, {len(param_maps)} param ({proc_high_conf} high-conf, {telemetry_boosted} tel-boosted)")

    total_attr_mappings = sum(len(p.get('attribute_lineage', [])) for p in procedures)

    # Calculate mapping type breakdown
    mapping_types = {}
    telemetry_confirmed_count = 0
    for proc in procedures:
        for m in proc.get('attribute_lineage', []):
            mtype = m.get('mapping_type', 'UNKNOWN')
            mapping_types[mtype] = mapping_types.get(mtype, 0) + 1
            if m.get('telemetry_confirmed'):
                telemetry_confirmed_count += 1

    # Build output
    parse_output = {
        'phase': 'PARSE',
        'timestamp': datetime.now().isoformat(),
        'version': '4.2',
        'stats': {
            'tables': ddl_parser.stats['tables'],
            'columns': ddl_parser.stats['columns'],
            'procedures': len(procedures),
            'attribute_mappings': total_attr_mappings,
            'parameter_mappings': total_param_mappings,
            'high_confidence_mappings': high_conf_count,
            'telemetry_confirmed_mappings': telemetry_confirmed_count,
            'imperva_logs': len(imperva_parser.logs),
            'mapping_types': mapping_types
        },
        'catalog': catalog,
        'procedures': procedures,
        'imperva': imperva_parser.to_dict()
    }

    # Save output
    with open(config.parse_output, 'w') as f:
        json.dump(parse_output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("PARSE AGENT COMPLETE (v4.2)")
    print("=" * 70)
    print("\nStats:")
    print(f"  Tables:                    {ddl_parser.stats['tables']}")
    print(f"  Columns:                   {ddl_parser.stats['columns']}")
    print(f"  Procedures:                {len(procedures)}")
    print(f"  Attribute Mappings:        {total_attr_mappings}")
    print(f"  Parameter Mappings:        {total_param_mappings}")
    print(f"  High Confidence (>=90%):   {high_conf_count}")
    print(f"  Telemetry Confirmed:       {telemetry_confirmed_count}")
    print(f"  Imperva Logs:              {len(imperva_parser.logs)}")
    print("\nMapping Types:")
    for mtype, count in sorted(mapping_types.items(), key=lambda x: -x[1]):
        print(f"    {mtype}: {count}")
    print(f"\nOutput: {config.parse_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
