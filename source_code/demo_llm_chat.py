#!/usr/bin/env python3
"""
demo_llm_chat.py - Chat over Lineage RAG with reranking

This script demonstrates how to:
1. Load the RAG binary produced by demo.py.
2. Perform stage-1 retrieval via LineageRAG.search().
3. Apply a stage-2 reranker using document metadata & query hints.
4. Build a compact context string for an LLM.
5. Call Anthropic Claude API for final answer synthesis.

Requirements:
- pip install anthropic
- Set ANTHROPIC_API_KEY environment variable

Fallback: If anthropic is not installed or API key is missing, uses simulated mode.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

from demo import LineageRAG

# Try to import Anthropic client
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# RAG binary path - relative to this script's parent directory
RAG_BINARY = Path(__file__).resolve().parent.parent / "output_artifacts" / "lineage_rag.pkl"

# Anthropic configuration
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # or "claude-3-haiku-20240307" for faster/cheaper
MAX_TOKENS = 1024


def extract_hints(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract simple hints:
      - table.column => (table, column, None)
      - mention of a procedure-like token => (None, None, proc)
    """
    q = query.lower()
    m = re.search(r"([a-z0-9_]+)\.([a-z0-9_]+)", q)
    table = col = proc = None
    if m:
        table, col = m.group(1), m.group(2)
    # crude heuristic for procedure names ending with _prs/_load/_enq
    pm = re.search(r"\b([a-z_][a-z0-9_]+_(?:prs|load|enq))\b", q)
    if pm:
        proc = pm.group(1)
    return table, col, proc


def rerank(results: List[Dict[str,Any]], query: str) -> List[Dict[str,Any]]:
    """
    Stage-2 reranker on top of RAG search/query results.

    Uses:
      - document type (attribute_lineage, table_lineage, parameter_lineage)
      - query hints (table/column/proc mentions)
      - telemetry contribution
    """
    table_hint, col_hint, proc_hint = extract_hints(query)
    ql = query.lower()

    def bonus(doc: Dict[str,Any]) -> float:
        b = 0.0
        doc_type = doc.get("type")

        # If the user is asking "how / mapping / compute" => prefer attribute_lineage
        if any(k in ql for k in ("how", "map", "mapping", "compute", "computed")):
            if doc_type == "attribute_lineage":
                b += 0.3

        # If user asks about "table", "flow", "lineage" => prefer table_lineage
        if any(k in ql for k in ("table", "flow", "lineage", "upstream", "downstream")):
            if doc_type == "table_lineage":
                b += 0.25

        # Parameters explicitly mentioned => parameter_lineage
        if "@" in ql and doc_type == "parameter_lineage":
            b += 0.3

        tgt = doc.get("target_table") or doc.get("table")
        src = doc.get("source_table")
        if table_hint and (tgt == table_hint or src == table_hint):
            b += 0.3
        if col_hint and doc.get("target_column") == col_hint:
            b += 0.3
        if proc_hint and doc.get("procedure") == proc_hint:
            b += 0.2

        # Telemetry contribution indicates relevance in production
        b += 0.15 * float(doc.get("telemetry_contribution", 0.0))

        return b

    for d in results:
        d["rerank_score"] = d.get("score", d.get("score_stage1", 0.0)) + bonus(d)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def format_context(results: List[Dict[str,Any]], max_docs: int = 6) -> str:
    """
    Build a compact text snippet from top-k reranked docs to feed to LLM.
    """
    lines = []
    for i, d in enumerate(results[:max_docs], start=1):
        t = d.get("type")
        txt = d.get("text","").strip()
        lines.append(f"[{i}. {t}] {txt}")
    return "\n".join(lines)


def _get_anthropic_client() -> Optional[Any]:
    """Get Anthropic client if available and configured."""
    if not ANTHROPIC_AVAILABLE:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _build_system_prompt() -> str:
    """Build the system prompt for lineage Q&A."""
    return """You are a data lineage expert assistant. Your role is to answer questions about data lineage based ONLY on the provided context from a RAG (Retrieval-Augmented Generation) system.

Guidelines:
1. Answer ONLY using information from the provided context. Do not make up information.
2. When describing lineage, use clear arrows: source_table.column → target_table.column
3. Mention the stored procedure (via_procedure) that performs the transformation.
4. Include confidence scores and mapping types when relevant.
5. If the context doesn't contain enough information to answer, say so clearly.
6. Be concise but thorough. Use bullet points for clarity.

Lineage types in context:
- table_lineage: Shows which tables feed into other tables via procedures
- attribute_lineage: Shows column-to-column mappings with confidence scores
- parameter_lineage: Shows how procedure parameters (@param) map to table columns
- catalog: Table and column definitions
- procedure: Stored procedure metadata"""


def synthesize_answer_with_llm(query: str, context: str, use_simulation: bool = False) -> str:
    """
    Synthesize an answer using Anthropic Claude API.

    Args:
        query: The user's question about data lineage
        context: RAG-retrieved context containing relevant lineage information
        use_simulation: Force simulation mode even if API is available

    Returns:
        LLM-generated answer grounded in the provided context

    Falls back to simulation if:
        - anthropic package is not installed
        - ANTHROPIC_API_KEY environment variable is not set
        - use_simulation=True is passed
    """
    client = _get_anthropic_client()

    if client is None or use_simulation:
        # Fallback to simulated response
        return f"""[SIMULATED LLM ANSWER - Set ANTHROPIC_API_KEY to enable real responses]

User question:
  {query}

Relevant lineage context:
{context}

Based on the retrieved lineage:

- The tables, columns, and procedures above describe how data flows.
- Attribute mappings (source_table.source_column → target_table.target_column) are
  the deterministic basis for attribute-level lineage.
- Parameter mappings (e.g. @param → table.column) indicate how filters and keys
  constrain the flow in the stored procedures.
- Telemetry scores (when present) indicate how often particular edges are exercised
  in production, which can be used to prioritize which lineage paths matter most.

To enable real LLM responses:
  1. pip install anthropic
  2. export ANTHROPIC_API_KEY=your_api_key
"""

    # Build the user message with context
    user_message = f"""Based on the following data lineage context, please answer the question.

=== LINEAGE CONTEXT ===
{context}
=== END CONTEXT ===

Question: {query}

Please provide a clear, concise answer based only on the information in the context above."""

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            system=_build_system_prompt(),
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract the text response
        answer = response.content[0].text
        return f"[Claude Response]\n\n{answer}"

    except anthropic.APIError as e:
        return f"[API Error] Failed to get response from Claude: {e}"
    except Exception as e:
        return f"[Error] Unexpected error during LLM call: {e}"


def main():
    print("\n" + "="*80)
    print("LINEAGE CHAT DEMO – RAG + RERANK + CLAUDE LLM")
    print("="*80)

    if not RAG_BINARY.exists():
        print(f"\n[ERROR] RAG binary not found at {RAG_BINARY}")
        print("   Run demo.py first to build output_artifacts/lineage_rag.pkl")
        return

    # Check LLM availability
    llm_client = _get_anthropic_client()
    if llm_client:
        print(f"\n[OK] Anthropic API configured (model: {ANTHROPIC_MODEL})")
        use_simulation = False
    else:
        if not ANTHROPIC_AVAILABLE:
            print("\n[WARN] anthropic package not installed. Run: pip install anthropic")
        else:
            print("\n[WARN] ANTHROPIC_API_KEY not set. Using simulated responses.")
        print("       To enable real LLM responses, set your API key:")
        print("       export ANTHROPIC_API_KEY=your_api_key")
        use_simulation = True

    print(f"\n[LOAD] Loading RAG from {RAG_BINARY}")
    rag = LineageRAG.load(RAG_BINARY)

    queries = [
        "Show me lineage for AAA_BORDEREAU_DETAILS_LOAD",
        "What tables does tnumgen write to?",
        "How is tlprdets.reserve_amt computed?",
        "Which columns are driven by @major_line_cd?",
        "Explain the data flow from toffprof to tlprdets"
    ]

    for i, q in enumerate(queries, start=1):
        print("\n" + "="*80)
        print(f"QUERY {i}: {q}")
        print("="*80)

        # Stage 1: baseline search (TF-IDF / embeddings + telemetry)
        print("\n🔍 STAGE 1 – Baseline RAG search")
        base_results = rag.search(q, k=10, boost_telemetry=True)
        for j, d in enumerate(base_results[:5], start=1):
            print(f"  {j}. [{d.get('type')}] score_stage1={d['score_stage1']:.3f} "
                  f"(semantic={d['semantic_score']:.3f}, "
                  f"telemetry+meta={d['telemetry_contribution']:.3f})")
            print(f"     {d['text'][:160]}...")

        # Stage 2: rerank by metadata & hints
        print("\n🎯 STAGE 2 – Reranking by table/column/proc/parameter hints")
        reranked = rerank(base_results, q)
        for j, d in enumerate(reranked[:5], start=1):
            print(f"  {j}. [{d.get('type')}] rerank_score={d['rerank_score']:.3f} "
                  f"(stage1={d['score_stage1']:.3f})")
            print(f"     {d['text'][:160]}...")

        # Stage 3: build context and call LLM
        mode_str = "simulated" if use_simulation else "Claude API"
        print(f"\n[LLM] STAGE 3 – Synthesized answer ({mode_str})")
        context = format_context(reranked, max_docs=6)
        answer = synthesize_answer_with_llm(q, context, use_simulation=use_simulation)
        print(answer)

    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    if use_simulation:
        print("\nTo enable real Claude responses:")
        print("  1. pip install anthropic")
        print("  2. export ANTHROPIC_API_KEY=your_api_key")
        print("  3. Run this script again\n")
    else:
        print("\nClaude API responses enabled. All answers were generated by Claude.\n")


if __name__ == "__main__":
    main()
