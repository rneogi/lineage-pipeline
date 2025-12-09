"""
abstract_agent.py - PHASE 2: ABSTRACT Agent (v4.2)

Builds YAML Knowledge Graph from parsed data with Imperva telemetry annotations.
Reads from Parse Agent output and produces enriched YAML KG.

Features (v4.2):
- Table-level lineage (source -> target via procedure)
- Attribute-level lineage (column-to-column mappings)
- Parameter-level lineage (@param -> table.column)
- Telemetry-enriched confidence scores

Usage:
    python abstract_agent.py

Input:
    intermediate/parse_output.json - Output from Parse Agent

Outputs:
    output_artifacts/demo.enriched.yaml - YAML Knowledge Graph with telemetry
    intermediate/abstract_output.json - JSON summary for next stage
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Input paths
    parse_output: Path = field(init=False)

    # Output paths
    intermediate_dir: Path = field(init=False)
    output_artifacts_dir: Path = field(init=False)
    yaml_output: Path = field(init=False)
    abstract_output: Path = field(init=False)

    def __post_init__(self):
        self.intermediate_dir = self.base_dir / "intermediate"
        self.output_artifacts_dir = self.base_dir / "output_artifacts"

        self.parse_output = self.intermediate_dir / "parse_output.json"
        self.yaml_output = self.output_artifacts_dir / "demo.enriched.yaml"
        self.abstract_output = self.intermediate_dir / "abstract_output.json"

        # Create directories
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.output_artifacts_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TELEMETRY HELPER
# ============================================================================

class TelemetryHelper:
    """Helper class to compute telemetry scores from Imperva data"""

    def __init__(self, telemetry_data: Dict[str, Any]):
        self.table_access_freq = telemetry_data.get('table_access_freq', {})
        self.procedure_call_freq = telemetry_data.get('procedure_call_freq', {})
        self.table_proc_cooc = telemetry_data.get('table_proc_cooc', {})

        # Compute max values for normalization
        self.max_table_freq = max(self.table_access_freq.values()) if self.table_access_freq else 1
        self.max_proc_freq = max(self.procedure_call_freq.values()) if self.procedure_call_freq else 1

    def get_table_importance_score(self, table: str) -> float:
        """Get normalized importance score for a table based on access frequency"""
        freq = self.table_access_freq.get(table.lower(), 0)
        return freq / self.max_table_freq if self.max_table_freq > 0 else 0.0

    def get_procedure_importance_score(self, procedure: str) -> float:
        """Get normalized importance score for a procedure based on call frequency"""
        freq = self.procedure_call_freq.get(procedure.lower(), 0)
        return freq / self.max_proc_freq if self.max_proc_freq > 0 else 0.0


# ============================================================================
# YAML KNOWLEDGE GRAPH BUILDER
# ============================================================================

class YAMLKnowledgeGraph:
    """Build structured YAML representation of lineage with deterministic mappings and telemetry"""

    def __init__(self):
        self.graph = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '4.2',
                'source': 'Iberia Legacy System',
                'features': [
                    'Deterministic attribute lineage',
                    'Parameter lineage extraction',
                    'Telemetry-enriched confidence scores',
                    'No cross-product mappings'
                ]
            },
            'catalog': {},
            'procedures': [],
            'lineage': {
                'table_level': [],
                'attribute_level': [],
                'parameter_level': []
            },
            'runtime_stats': {},
            'telemetry': {}
        }

    def build_from_parsed_data(
        self,
        catalog: Dict[str, List[Dict]],
        procedures: List[Dict[str, Any]],
        telemetry_helper: Optional[TelemetryHelper] = None,
        telemetry_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build complete YAML knowledge graph with telemetry annotations"""

        # Add catalog
        self.graph['catalog'] = {
            table: [{
                'name': col['name'],
                'type': col['type'],
                'nullable': col['nullable']
            } for col in cols]
            for table, cols in catalog.items()
        }

        # Add procedures and build lineage
        for proc in procedures:
            self.graph['procedures'].append({
                'name': proc['name'],
                'tables': proc['all_tables'],
                'source_tables': proc['source_tables'],
                'target_tables': proc['target_tables'],
                'attribute_mappings_count': len(proc.get('attribute_lineage', [])),
                'parameter_mappings_count': len(proc.get('parameter_lineage', []))
            })

            # Build table-level lineage
            for src in proc['source_tables']:
                for tgt in proc['target_tables']:
                    edge = {
                        'source': src,
                        'target': tgt,
                        'via_procedure': proc['name']
                    }

                    # Add telemetry if available
                    if telemetry_helper:
                        src_score = telemetry_helper.get_table_importance_score(src)
                        tgt_score = telemetry_helper.get_table_importance_score(tgt)
                        proc_score = telemetry_helper.get_procedure_importance_score(proc['name'])
                        edge['telemetry_score'] = (src_score + tgt_score + proc_score) / 3
                    else:
                        edge['telemetry_score'] = 0.0

                    self.graph['lineage']['table_level'].append(edge)

            # Add DETERMINISTIC attribute-level lineage (no cross-product!)
            for mapping in proc.get('attribute_lineage', []):
                attr_edge = {
                    'source_table': mapping['source_table'],
                    'source_column': mapping['source_column'],
                    'target_table': mapping['target_table'],
                    'target_column': mapping['target_column'],
                    'via_procedure': mapping['via_procedure'],
                    'confidence': mapping['confidence'],
                    'mapping_type': mapping['mapping_type']
                }

                # Enrich with telemetry
                if telemetry_helper:
                    src_score = telemetry_helper.get_table_importance_score(mapping['source_table'])
                    tgt_score = telemetry_helper.get_table_importance_score(mapping['target_table'])
                    # Boost confidence for frequently accessed tables
                    telemetry_boost = (src_score + tgt_score) / 2 * 0.05  # Up to 5% boost
                    attr_edge['confidence'] = min(0.99, attr_edge['confidence'] + telemetry_boost)
                    attr_edge['telemetry_score'] = (src_score + tgt_score) / 2
                else:
                    attr_edge['telemetry_score'] = 0.0

                self.graph['lineage']['attribute_level'].append(attr_edge)

            # Add PARAMETER-level lineage
            for pm in proc.get('parameter_lineage', []):
                param_edge = {
                    'parameter': pm['parameter'],
                    'table': pm['table'],
                    'column': pm['column'],
                    'via_procedure': pm['via_procedure'],
                    'pattern': pm['pattern'],
                    'confidence': pm['confidence']
                }

                # Enrich with telemetry
                if telemetry_helper and pm['table']:
                    table_score = telemetry_helper.get_table_importance_score(pm['table'])
                    param_edge['telemetry_score'] = table_score
                else:
                    param_edge['telemetry_score'] = 0.0

                self.graph['lineage']['parameter_level'].append(param_edge)

        # Add telemetry statistics
        if telemetry_data:
            self.graph['telemetry'] = {
                'table_access_frequency': telemetry_data.get('table_access_freq', {}),
                'procedure_call_frequency': telemetry_data.get('procedure_call_freq', {}),
                'total_log_entries': len(telemetry_data.get('sql_patterns', []))
            }

        return self.graph

    def save_yaml(self, path: Path):
        """Save YAML to file"""
        with open(path, 'w') as f:
            yaml.safe_dump(self.graph, f, sort_keys=False, default_flow_style=False)
        print(f"Saved YAML to {path}")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the knowledge graph"""
        return {
            'tables': len(self.graph['catalog']),
            'procedures': len(self.graph['procedures']),
            'table_edges': len(self.graph['lineage']['table_level']),
            'attribute_edges': len(self.graph['lineage']['attribute_level']),
            'parameter_edges': len(self.graph['lineage']['parameter_level'])
        }


# ============================================================================
# MAIN ABSTRACT AGENT
# ============================================================================

def main():
    """Execute Abstract Agent"""
    print("=" * 70)
    print("PHASE 2: ABSTRACT AGENT")
    print("=" * 70)
    print("Building YAML Knowledge Graph with Imperva telemetry annotations")
    print("-" * 70)

    config = Config()

    # Load Parse Agent output
    print(f"\nLoading Parse output from {config.parse_output}...")
    if not config.parse_output.exists():
        print(f"ERROR: Parse output not found: {config.parse_output}")
        print("Please run parse_agent.py first.")
        return

    with open(config.parse_output, 'r') as f:
        parse_data = json.load(f)

    print(f"Loaded: {parse_data['stats']['tables']} tables, "
          f"{parse_data['stats']['procedures']} procedures, "
          f"{parse_data['stats']['imperva_logs']} telemetry logs")

    # Extract data
    catalog = parse_data['catalog']
    procedures = parse_data['procedures']
    imperva_data = parse_data.get('imperva', {})
    telemetry = imperva_data.get('telemetry', {})

    # Create telemetry helper if data available
    telemetry_helper = None
    if telemetry and telemetry.get('table_access_freq'):
        telemetry_helper = TelemetryHelper(telemetry)
        print(f"\nTelemetry enabled: {len(telemetry['table_access_freq'])} tables, "
              f"{len(telemetry['procedure_call_freq'])} procedures tracked")
    else:
        print("\nNo telemetry data available - building graph without enrichment")

    # Build Knowledge Graph
    print("\nBuilding YAML Knowledge Graph...")
    kg = YAMLKnowledgeGraph()
    yaml_graph = kg.build_from_parsed_data(
        catalog=catalog,
        procedures=procedures,
        telemetry_helper=telemetry_helper,
        telemetry_data=telemetry
    )

    # Save YAML
    kg.save_yaml(config.yaml_output)

    # Get stats
    stats = kg.get_stats()

    # Save abstract output for next stage
    abstract_output = {
        'phase': 'ABSTRACT',
        'timestamp': datetime.now().isoformat(),
        'version': '4.2',
        'input_file': str(config.parse_output),
        'output_file': str(config.yaml_output),
        'stats': stats,
        'telemetry_enabled': telemetry_helper is not None,
        'yaml_graph': yaml_graph  # Include full graph for next stage
    }

    with open(config.abstract_output, 'w') as f:
        json.dump(abstract_output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("ABSTRACT AGENT COMPLETE (v4.2)")
    print("=" * 70)
    print("\nKnowledge Graph Statistics:")
    print(f"  Tables:           {stats['tables']}")
    print(f"  Procedures:       {stats['procedures']}")
    print(f"  Table Edges:      {stats['table_edges']}")
    print(f"  Attribute Edges:  {stats['attribute_edges']} (NO CROSS-PRODUCT!)")
    print(f"  Parameter Edges:  {stats['parameter_edges']}")
    print("\nOutputs:")
    print(f"  YAML:  {config.yaml_output}")
    print(f"  JSON:  {config.abstract_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
