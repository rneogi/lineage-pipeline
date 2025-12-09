"""
retrieve_agent.py - PHASE 3: RETRIEVE Agent (v5.0)

Builds RAG (Retrieval-Augmented Generation) library with ChromaDB and ZeroEntropy.

Features (v5.0):
- ChromaDB as persistent vector store
- ZeroEntropy reranker for stage-2 reranking
- Semantic embeddings via Chroma's default embedding function
- Parameter lineage documents in RAG index
- Telemetry-boosted scoring

Usage:
    python retrieve_agent.py

Input:
    intermediate/abstract_output.json - Output from Abstract Agent

Outputs:
    output_artifacts/lineage_rag.pkl - RAG metadata (documents)
    chroma_db/ - ChromaDB persistent vector store
    intermediate/retrieve_output.json - JSON summary for next stage
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import Counter

# Check for ChromaDB
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    ChromaSettings = None
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Install with: pip install chromadb")

# Check for ZeroEntropy
try:
    from zeroentropy import ZeroEntropy
    ZEROENTROPY_AVAILABLE = True
except ImportError:
    ZeroEntropy = None
    ZEROENTROPY_AVAILABLE = False
    print("Warning: zeroentropy not installed. Stage-2 reranking disabled.")

# Check for OpenAI (embeddings)
import os
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Pipeline configuration"""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Input paths
    abstract_output: Path = field(init=False)

    # Output paths
    intermediate_dir: Path = field(init=False)
    output_artifacts_dir: Path = field(init=False)
    rag_output: Path = field(init=False)
    retrieve_output: Path = field(init=False)
    chroma_path: Path = field(init=False)

    def __post_init__(self):
        self.intermediate_dir = self.base_dir / "intermediate"
        self.output_artifacts_dir = self.base_dir / "output_artifacts"

        self.abstract_output = self.intermediate_dir / "abstract_output.json"
        self.rag_output = self.output_artifacts_dir / "lineage_rag.pkl"
        self.retrieve_output = self.intermediate_dir / "retrieve_output.json"
        self.chroma_path = self.base_dir / "chroma_db"

        # Create directories
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.output_artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LINEAGE RAG with ChromaDB + ZeroEntropy
# ============================================================================

class LineageRAG:
    """
    Retrieval-Augmented Generation system using ChromaDB + ZeroEntropy.

    v5.0 Features:
    - ChromaDB as persistent vector store
    - ZeroEntropy reranker for stage-2 reranking
    - Telemetry-boosted scoring
    - Full compatibility with chat_interface.py
    """

    def __init__(self, chroma_path: Path = None):
        if not CHROMA_AVAILABLE:
            raise RuntimeError(
                "ChromaDB is required but not installed. "
                "Install with: pip install chromadb"
            )

        self.documents: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "telemetry_enabled": False,
            "vector_store": "chroma",
        }

        # Chroma setup
        self.chroma_path = chroma_path or Path("chroma_db")
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = "lineage_collection"

        # Initialize OpenAI embedding function
        self._openai_client = None
        self._embedding_function = None
        self._init_openai_embedding()
        self._init_chroma()

        # ZeroEntropy reranker client
        self.zclient = None
        if ZEROENTROPY_AVAILABLE:
            try:
                self.zclient = ZeroEntropy()
                print("✓ ZeroEntropy reranker initialized")
            except Exception as e:
                print(f"⚠ ZeroEntropy init failed: {e}")
                self.zclient = None

    def _init_openai_embedding(self):
        """Initialize OpenAI client and embedding function if OPENAI_API_KEY is set."""
        if not OPENAI_AVAILABLE:
            print("⚠ openai package not installed. Install with: pip install openai")
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠ OPENAI_API_KEY not set. Chroma collection will exist but cannot embed remotely.")
            return

        self._openai_client = OpenAI(api_key=api_key)

        class _OpenAIEmbeddingFunction:
            def __init__(self, client, model: str = "text-embedding-3-small"):
                self.client = client
                self.model = model

            def __call__(self, input: List[str]) -> List[List[float]]:
                resp = self.client.embeddings.create(model=self.model, input=input)
                return [d.embedding for d in resp.data]

            def name(self) -> str:
                return "openai-text-embedding-3-small"

            def embed_query(self, input: List[str]) -> List[List[float]]:
                """Required by ChromaDB for query embedding"""
                resp = self.client.embeddings.create(model=self.model, input=input)
                return [d.embedding for d in resp.data]

        self._embedding_function = _OpenAIEmbeddingFunction(self._openai_client)
        self.metadata["embedding_model"] = "text-embedding-3-small (openai)"
        print("✓ OpenAI embedding function initialized")

    def _init_chroma(self):
        """Initialize ChromaDB client and collection"""
        settings = ChromaSettings() if ChromaSettings else None
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=settings,
        )
        # Delete existing collection to start fresh
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
        # Use OpenAI embedding if available, otherwise use ChromaDB's default local model
        if self._embedding_function is not None:
            self.collection = self.chroma_client.get_or_create_collection(
                self.collection_name,
                embedding_function=self._embedding_function,
            )
        else:
            # Use ChromaDB's default local embedding (all-MiniLM-L6-v2)
            self.collection = self.chroma_client.get_or_create_collection(
                self.collection_name,
            )
            self.metadata["embedding_model"] = "all-MiniLM-L6-v2 (local)"
            print("✓ Using ChromaDB's default local embedding model")
        self.metadata["chroma_collection"] = self.collection_name

    def add_documents_from_yaml(self, yaml_data: Dict[str, Any]):
        """
        Convert KG YAML into RAG documents and store them in Chroma.
        """
        tel = yaml_data.get("telemetry", {})
        t_freq = tel.get("table_access_frequency", {}) or {}
        p_freq = tel.get("procedure_call_frequency", {}) or {}
        self.metadata["telemetry_enabled"] = bool(t_freq or p_freq)
        max_t = max(t_freq.values()) if t_freq else 1
        max_p = max(p_freq.values()) if p_freq else 1

        def _add(doc: Dict[str, Any]):
            # ChromaDB will use its default local embedding if _embedding_function is None
            text = doc["text"]
            meta = {k: v for k, v in doc.items() if k != "text"}
            doc_id = meta.get("id")
            if not doc_id:
                base = meta.get("type", "doc")
                suffix = len(self.documents)
                doc_id = f"{base}:{suffix}"
                meta["id"] = doc_id

            # Convert non-string metadata to strings for Chroma
            meta_clean = {}
            for k, v in meta.items():
                if isinstance(v, (list, dict)):
                    meta_clean[k] = json.dumps(v)
                elif isinstance(v, (int, float, bool)):
                    meta_clean[k] = v
                else:
                    meta_clean[k] = str(v)

            self.documents.append(doc)
            self.collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta_clean],
            )

        # Catalog documents
        for t, cols in yaml_data.get("catalog", {}).items():
            cns = [c["name"] for c in cols]
            cts = [c["type"] for c in cols]
            score = t_freq.get(t, 0) / max_t if max_t > 0 else 0.0
            doc = {
                "type": "catalog",
                "table": t,
                "text": f"Table {t} columns: {', '.join(cns)} types: {', '.join(cts)}",
                "telemetry_score": score,
            }
            _add(doc)

        # Procedure documents
        for p in yaml_data.get("procedures", []):
            name = p["name"]
            sc = p_freq.get(name.lower(), 0) / max_p if max_p > 0 else 0.0
            doc = {
                "type": "procedure",
                "procedure": name,
                "source_tables": p["source_tables"],
                "target_tables": p["target_tables"],
                "text": (
                    f"Procedure {name} reads {', '.join(p['source_tables'])} "
                    f"and writes {', '.join(p['target_tables'])}"
                ),
                "telemetry_score": sc,
            }
            _add(doc)

        # Table-level lineage
        for e in yaml_data.get("lineage", {}).get("table_level", []):
            doc = {
                "type": "table_lineage",
                "source_table": e["source"],
                "target_table": e["target"],
                "procedure": e["via_procedure"],
                "text": (
                    f"Table lineage: {e['source']} -> {e['target']} "
                    f"via {e['via_procedure']}"
                ),
                "telemetry_score": e.get("telemetry_score", 0.0),
            }
            _add(doc)

        # Attribute-level lineage
        for e in yaml_data.get("lineage", {}).get("attribute_level", []):
            doc = {
                "type": "attribute_lineage",
                "source_table": e["source_table"],
                "source_column": e["source_column"],
                "target_table": e["target_table"],
                "target_column": e["target_column"],
                "procedure": e["via_procedure"],
                "confidence": e.get("confidence", 0.75),
                "mapping_type": e.get("mapping_type", "UNKNOWN"),
                "text": (
                    f"Attribute lineage: {e['source_table']}.{e['source_column']} "
                    f"-> {e['target_table']}.{e['target_column']} "
                    f"via {e['via_procedure']} "
                    f"(confidence={e.get('confidence',0.75):.2f}, "
                    f"type={e.get('mapping_type','UNKNOWN')})"
                ),
                "telemetry_score": e.get("telemetry_score", 0.0),
            }
            _add(doc)

        # Parameter-level lineage
        for e in yaml_data.get("lineage", {}).get("parameter_level", []):
            doc = {
                "type": "parameter_lineage",
                "parameter": e["parameter"],
                "table": e["table"],
                "column": e["column"],
                "procedure": e["via_procedure"],
                "text": (
                    f"Parameter {e['parameter']} filters {e['table']}.{e['column']} "
                    f"in {e['via_procedure']} ({e.get('pattern', 'unknown')})"
                ),
                "confidence": e.get("confidence", 0.8),
                "telemetry_score": e.get("telemetry_score", 0.0),
            }
            _add(doc)

        # Add telemetry documents for direct querying
        # Top procedure call frequencies from Imperva logs
        if p_freq:
            sorted_procs = sorted(p_freq.items(), key=lambda x: x[1], reverse=True)
            for proc, calls in sorted_procs[:50]:  # Top 50 procedures
                doc = {
                    "type": "telemetry_procedure",
                    "procedure": proc,
                    "call_count": calls,
                    "normalized_score": calls / max_p if max_p > 0 else 0.0,
                    "text": (
                        f"Telemetry: Procedure {proc} was called {calls} times "
                        f"(frequency rank: top {sorted_procs.index((proc, calls)) + 1})"
                    ),
                }
                _add(doc)

        # Top table access frequencies from Imperva logs
        if t_freq:
            sorted_tables = sorted(t_freq.items(), key=lambda x: x[1], reverse=True)
            for table, accesses in sorted_tables[:50]:  # Top 50 tables
                doc = {
                    "type": "telemetry_table",
                    "table": table,
                    "access_count": accesses,
                    "normalized_score": accesses / max_t if max_t > 0 else 0.0,
                    "text": (
                        f"Telemetry: Table {table} was accessed {accesses} times "
                        f"(access rank: top {sorted_tables.index((table, accesses)) + 1})"
                    ),
                }
                _add(doc)

        # Create hotspot analysis summary document
        if p_freq:
            top_5_procs = sorted(p_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            hotspot_text = "Hotspot analysis from Imperva telemetry:\n"
            hotspot_text += "Top procedures by call frequency:\n"
            for i, (proc, calls) in enumerate(top_5_procs, 1):
                hotspot_text += f"  {i}. {proc}: {calls} calls\n"
            doc = {
                "type": "hotspot_summary",
                "text": hotspot_text,
                "top_procedures": [p[0] for p in top_5_procs],
            }
            _add(doc)

        # =====================================================================
        # PRE-COMPUTED STATISTICS DOCUMENTS
        # These provide accurate aggregate answers for analytical queries
        # =====================================================================

        # Compute table participation statistics from lineage edges
        # Note: yaml_data here is already the yaml_graph content (passed from main)
        # So lineage is directly accessible, not nested under yaml_graph
        lineage_data = yaml_data.get("lineage", {})
        table_level = lineage_data.get("table_level", [])
        if table_level:
            source_counts = Counter()
            target_counts = Counter()
            total_counts = Counter()

            for edge in table_level:
                src = edge.get("source", "").lower()
                tgt = edge.get("target", "").lower()
                if src:
                    source_counts[src] += 1
                    total_counts[src] += 1
                if tgt:
                    target_counts[tgt] += 1
                    total_counts[tgt] += 1

            # Document: Top tables by total lineage participation
            top_15_total = total_counts.most_common(15)
            participation_text = f"Table Participation Statistics (from {len(table_level)} lineage edges):\n\n"
            participation_text += "Top 15 Tables by Total Participation in SP Read/Write Flows:\n"
            for i, (table, count) in enumerate(top_15_total, 1):
                src = source_counts.get(table, 0)
                tgt = target_counts.get(table, 0)
                participation_text += f"  {i}. {table}: {count} total (as source/read: {src}, as target/write: {tgt})\n"

            doc = {
                "type": "lineage_statistics",
                "subtype": "table_participation",
                "text": participation_text,
                "top_tables": [t[0] for t in top_15_total],
                "total_edges": len(table_level),
            }
            _add(doc)

            # Document: Top tables by read frequency (source)
            top_10_source = source_counts.most_common(10)
            source_text = "Tables Most Frequently READ FROM (as source in lineage):\n"
            for i, (table, count) in enumerate(top_10_source, 1):
                source_text += f"  {i}. {table}: {count} read operations\n"

            doc = {
                "type": "lineage_statistics",
                "subtype": "most_read_tables",
                "text": source_text,
                "top_tables": [t[0] for t in top_10_source],
            }
            _add(doc)

            # Document: Top tables by write frequency (target)
            top_10_target = target_counts.most_common(10)
            target_text = "Tables Most Frequently WRITTEN TO (as target in lineage):\n"
            for i, (table, count) in enumerate(top_10_target, 1):
                target_text += f"  {i}. {table}: {count} write operations\n"

            doc = {
                "type": "lineage_statistics",
                "subtype": "most_written_tables",
                "text": target_text,
                "top_tables": [t[0] for t in top_10_target],
            }
            _add(doc)

            # Document: Performance bottleneck analysis
            # Tables with high bidirectional activity are potential bottlenecks
            bottleneck_candidates = []
            for table in total_counts:
                src = source_counts.get(table, 0)
                tgt = target_counts.get(table, 0)
                if src > 0 and tgt > 0:  # Bidirectional
                    bottleneck_candidates.append({
                        "table": table,
                        "total": total_counts[table],
                        "reads": src,
                        "writes": tgt,
                        "bidirectional_score": min(src, tgt) / max(src, tgt) if max(src, tgt) > 0 else 0
                    })

            # Sort by total participation and bidirectional score
            bottleneck_candidates.sort(key=lambda x: (x["total"], x["bidirectional_score"]), reverse=True)

            bottleneck_text = "Performance Bottleneck Analysis (tables with high bidirectional read/write activity):\n\n"
            bottleneck_text += "These tables are both read from AND written to frequently, indicating potential lock contention:\n\n"
            for i, b in enumerate(bottleneck_candidates[:10], 1):
                bottleneck_text += f"  {i}. {b['table']}: {b['total']} total operations "
                bottleneck_text += f"(reads: {b['reads']}, writes: {b['writes']})\n"

            if bottleneck_candidates:
                top_bottleneck = bottleneck_candidates[0]
                bottleneck_text += f"\nHighest risk: {top_bottleneck['table']} with {top_bottleneck['total']} operations "
                bottleneck_text += f"({top_bottleneck['reads']} reads, {top_bottleneck['writes']} writes)"

            doc = {
                "type": "lineage_statistics",
                "subtype": "bottleneck_analysis",
                "text": bottleneck_text,
                "bottleneck_tables": [b["table"] for b in bottleneck_candidates[:10]],
            }
            _add(doc)

            print(f"  Added {4} pre-computed statistics documents")

        # Stats
        by_type = Counter(d["type"] for d in self.documents)
        self.metadata["by_type"] = dict(by_type)
        self.metadata["doc_count"] = len(self.documents)

        print(f"Created {len(self.documents)} RAG documents (telemetry: {'enabled' if self.metadata['telemetry_enabled'] else 'disabled'})")

    def build(self):
        """Kept for compatibility; Chroma indexes on insert."""
        pass

    def search(
        self,
        query_text: str,
        k: int = 20,
        doc_type: Optional[str] = None,
        boost_telemetry: bool = True,
        telemetry_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Stage-1 retrieval via Chroma vector search."""
        if not self.documents:
            raise RuntimeError("No documents in RAG index.")

        n_results = max(k * 2, 100)

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        candidates: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            text = docs[i]
            meta = metas[i] or {}

            # distance -> similarity-ish score in [0,1)
            dist = float(dists[i]) if i < len(dists) else 0.0
            semantic_score = 1.0 / (1.0 + dist)

            telemetry_score = float(meta.get("telemetry_score", 0.0)) if boost_telemetry else 0.0
            score_stage1 = (1 - telemetry_weight) * semantic_score + telemetry_weight * telemetry_score

            d = dict(meta)
            d.setdefault("id", doc_id)
            d.setdefault("type", meta.get("type", "unknown"))
            d["text"] = text
            d["semantic_score"] = semantic_score
            d["telemetry_contribution"] = telemetry_score
            d["score_stage1"] = score_stage1

            if doc_type and d.get("type") != doc_type:
                continue

            candidates.append(d)
            if len(candidates) >= n_results:
                break

        return candidates

    def _rerank_with_zeroentropy(
        self,
        query_text: str,
        candidates: List[Dict[str, Any]],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Stage-2 rerank using ZeroEntropy if available; otherwise fallback."""
        if not self.zclient or not candidates:
            for c in candidates:
                c["score"] = c.get("score_stage1", 0.0)
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates[:k]

        documents = [c["text"] for c in candidates]
        try:
            response = self.zclient.models.rerank(
                model="zerank-2",
                query=query_text,
                documents=documents,
                top_n=min(k, len(documents)),
            )
            idx_to_score = {r.index: float(r.relevance_score) for r in response.results}

            for idx, c in enumerate(candidates):
                rerank_score = idx_to_score.get(idx, 0.0)
                c["zerank_score"] = rerank_score
                c["score"] = 0.5 * c.get("score_stage1", 0.0) + 0.5 * rerank_score

            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates[:k]

        except Exception as e:
            print(f"⚠ ZeroEntropy rerank failed: {e}. Falling back to stage1 scores.")
            for c in candidates:
                c["score"] = c.get("score_stage1", 0.0)
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return candidates[:k]

    def query(
        self,
        query_text: str,
        k: int = 10,
        doc_type: Optional[str] = None,
        boost_telemetry: bool = True,
        telemetry_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Full retrieval: stage-1 + optional ZeroEntropy rerank."""
        stage1 = self.search(
            query_text=query_text,
            k=k,
            doc_type=doc_type,
            boost_telemetry=boost_telemetry,
            telemetry_weight=telemetry_weight,
        )
        return self._rerank_with_zeroentropy(query_text, stage1, k)

    def save(self, path: Path):
        """Save RAG metadata (documents + metadata). Chroma is persisted separately."""
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved RAG metadata: {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: Path, chroma_path: Path = None) -> "LineageRAG":
        """Load RAG metadata and reconnect to existing Chroma collection."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Determine chroma path relative to rag path
        if chroma_path is None:
            chroma_path = path.parent.parent / "chroma_db"

        rag = cls(chroma_path=chroma_path)
        rag.documents = data.get("documents", [])
        rag.metadata = data.get("metadata", {})
        return rag


# ============================================================================
# MAIN RETRIEVE AGENT
# ============================================================================

def main():
    """Execute Retrieve Agent"""
    print("=" * 70)
    print("PHASE 3: RETRIEVE AGENT (v5.0)")
    print("=" * 70)
    print("Building RAG Library with ChromaDB + ZeroEntropy Reranking")
    print("-" * 70)

    config = Config()

    # Load Abstract Agent output
    print(f"\nLoading Abstract output from {config.abstract_output}...")
    if not config.abstract_output.exists():
        print(f"ERROR: Abstract output not found: {config.abstract_output}")
        print("Please run abstract_agent.py first.")
        return

    with open(config.abstract_output, 'r') as f:
        abstract_data = json.load(f)

    yaml_graph = abstract_data['yaml_graph']
    print(f"Loaded: {abstract_data['stats']['tables']} tables, "
          f"{abstract_data['stats']['procedures']} procedures")

    # Build RAG with ChromaDB
    print("\nBuilding RAG with ChromaDB...")
    rag = LineageRAG(chroma_path=config.chroma_path)
    rag.add_documents_from_yaml(yaml_graph)
    rag.build()  # no-op for Chroma

    # Save RAG metadata
    rag.save(config.rag_output)

    # Test query
    print("\nTesting retrieval:")
    test_query = "show me frequently accessed tables"
    results = rag.query(test_query, k=5, boost_telemetry=True)

    print(f"\n  Query: '{test_query}'")
    for i, result in enumerate(results[:3], 1):
        score_info = f"score={result.get('score', 0):.3f}"
        if 'zerank_score' in result:
            score_info += f" (zerank={result['zerank_score']:.3f})"
        print(f"  {i}. [{result['type']}] {score_info}")
        print(f"     {result['text'][:80]}...")

    # Save retrieve output for next stage
    retrieve_output = {
        'phase': 'RETRIEVE',
        'timestamp': datetime.now().isoformat(),
        'version': '5.0',
        'input_file': str(config.abstract_output),
        'output_file': str(config.rag_output),
        'stats': {
            'documents': len(rag.documents),
            'vector_store': 'chromadb',
            'reranker': 'zeroentropy' if rag.zclient else 'none',
            'telemetry_enabled': rag.metadata.get('telemetry_enabled'),
        },
        'rag_path': str(config.rag_output),
        'chroma_path': str(config.chroma_path),
        'yaml_graph': yaml_graph  # Pass through for Generate agent
    }

    with open(config.retrieve_output, 'w') as f:
        json.dump(retrieve_output, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("RETRIEVE AGENT COMPLETE (v5.0)")
    print("=" * 70)
    print("\nRAG Statistics:")
    print(f"  Documents:    {len(rag.documents)}")
    print(f"  Vector Store: ChromaDB")
    print(f"  Reranker:     {'ZeroEntropy' if rag.zclient else 'None (fallback)'}")
    print(f"  Telemetry:    {'enabled' if rag.metadata.get('telemetry_enabled') else 'disabled'}")
    print("\nOutputs:")
    print(f"  RAG Metadata: {config.rag_output}")
    print(f"  ChromaDB:     {config.chroma_path}")
    print(f"  JSON:         {config.retrieve_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
