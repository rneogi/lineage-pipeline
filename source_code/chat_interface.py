"""
chat_interface.py - Interactive Chat Interface for Lineage Queries with LLM (v5.0)

Wired to:
- lineage_rag.pkl (metadata) + Chroma vector store with OpenAI embeddings
- Optional ZeroEntropy reranker (inside LineageRAG)
- Optional ground-truth YAMLs (table.yaml, sp.yaml, log.yaml) for confidence estimation.

Usage:
    python chat_interface.py
    python chat_interface.py --no-llm  # RAG-only mode

"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

import yaml

from demo import LineageRAG

# Anthropic API for LLM
try:
    import anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def get_api_key() -> str:
    """Get Anthropic API key from environment or config file."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "api_key.txt"
    if config_path.exists():
        with open(config_path, "r") as f:
            api_key = f.read().strip()
            if api_key:
                return api_key

    home_config = Path.home() / ".anthropic_api_key"
    if home_config.exists():
        with open(home_config, "r") as f:
            api_key = f.read().strip()
            if api_key:
                return api_key

    return None


class GroundTruthValidator:
    """Simple per-query validator using available YAML files."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.sources: Dict[str, Dict[str, Any]] = {}

        # Check for ground-truth files in input_data/ or output_artifacts/
        candidates = {
            "table": base_dir / "input_data" / "table.yaml",
            "sp":    base_dir / "input_data" / "sp.yaml",
            "log":   base_dir / "input_data" / "log.yaml",
            "enriched": base_dir / "output_artifacts" / "demo.enriched.yaml",
        }

        for name, path in candidates.items():
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    self.sources[name] = {"path": path, "data": data}
                    print(f"✓ Ground truth loaded: {name} -> {path}")
                except Exception as e:
                    print(f"⚠ Failed to load {path}: {e}")

        if not self.sources:
            print(
                "⚠ No ground-truth YAMLs found. "
                "Confidence will rely on retrieval scores only.",
            )

    def assess(self, query: str, answer: str, rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rag_results:
            return {
                'score': 0.0,
                'bucket': 'low',
                'base': 0.0,
                'coverage': 0.0,
                'reason': 'no_rag_results',
            }

        # Get max retrieval score (combined score from RAG)
        max_score = max(float(r.get("score", r.get("score_stage1", 0.0))) for r in rag_results)

        # Get max zerank (reranker) score if available - this is the key relevance signal
        max_zerank = max((float(r.get("zerank_score", 0.0)) for r in rag_results), default=0.0)

        # Ground-truth coverage check
        coverage = 0.0
        if self.sources:
            combined_raw = ' '.join(str(v['data']) for v in self.sources.values())
            hits = 0
            total = 0
            for r in rag_results[:5]:
                for key in ('table', 'source_table', 'target_table', 'procedure'):
                    val = r.get(key)
                    if isinstance(val, str):
                        total += 1
                        if val in combined_raw:
                            hits += 1
            if total > 0:
                coverage = hits / total

        # Improved confidence formula:
        # - Use rerank score as primary relevance signal (it's more accurate than raw retrieval)
        # - If no reranker, fall back to scaled retrieval
        # - Ground-truth coverage is a secondary validation
        if max_zerank > 0:
            # Reranker available: weight rerank heavily (50%), coverage (30%), retrieval (20%)
            scaled_retrieval = min(1.0, max_score * 1.5)  # Less aggressive scaling
            final_score = 0.5 * max_zerank + 0.3 * coverage + 0.2 * scaled_retrieval
        else:
            # No reranker: use original formula but less aggressive scaling
            scaled_retrieval = min(1.0, max_score * 2.0)
            final_score = 0.5 * scaled_retrieval + 0.5 * coverage

        final_score = max(0.0, min(final_score, 1.0))

        # Adjusted thresholds for realistic scoring
        if final_score >= 0.6:
            bucket = 'high'
        elif final_score >= 0.4:
            bucket = 'medium'
        else:
            bucket = 'low'

        return {
            'score': final_score,
            'bucket': bucket,
            'base': scaled_retrieval,
            'coverage': coverage,
            'raw_retrieval': max_score,
            'zerank': max_zerank,
        }

    @staticmethod
    def format(result: Dict[str, Any]) -> str:
        raw = result.get('raw_retrieval', result.get('base', 0.0))
        zerank = result.get('zerank', 0.0)
        return (
            f"Confidence: {result['score']:.0%} ({result['bucket']})  ·  "
            f"retrieval={raw:.2f}, rerank={zerank:.2f}, "
            f"ground-truth={result.get('coverage', 0.0):.0%}"
        )


class LineageChatInterface:
    """Interactive chat interface for lineage queries with optional LLM synthesis (v5.0)."""

    def __init__(self, rag_path: Path, use_llm: bool = True):
        self.base_dir = rag_path.resolve().parent.parent
        self.rag = LineageRAG.load(rag_path)
        self.history: List[Dict[str, str]] = []
        self.use_llm = use_llm and LLM_AVAILABLE

        self.gt_validator = GroundTruthValidator(self.base_dir)

        self.client = None
        if self.use_llm:
            api_key = get_api_key()
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                print("LLM integration enabled (Claude)")
            else:
                print("API key not found. Using RAG-only mode.")
                self.use_llm = False

        docs = self.rag.documents
        self.stats = {
            'total_docs': len(docs),
            'catalog_tables': len([d for d in docs if d.get('type') == 'catalog']),
            'procedures': len([d for d in docs if d.get('type') == 'procedure']),
            'table_edges': len([d for d in docs if d.get('type') == 'table_lineage']),
            'attr_edges': len([d for d in docs if d.get('type') == 'attribute_lineage']),
            'param_edges': len([d for d in docs if d.get('type') == 'parameter_lineage']),
        }

    def welcome(self):
        print("\n" + "=" * 70)
        print("LINEAGE CHAT INTERFACE v5.0" + (" with LLM" if self.use_llm else " (RAG-only)"))
        print("=" * 70)
        print("\nKnowledge Base:")
        print(f"   - {self.stats['catalog_tables']} tables in catalog")
        print(f"   - {self.stats['procedures']} stored procedures")
        print(f"   - {self.stats['table_edges']} table lineage edges")
        print(f"   - {self.stats['attr_edges']} attribute lineage edges")
        print(f"   - {self.stats['param_edges']} parameter lineage edges")
        print("\nGround Truth YAMLs (for confidence): table.yaml / sp.yaml / log.yaml in input_data/ if present.")
        if self.use_llm:
            print("\nLLM: Enabled (Claude)")
        else:
            print("\nMode: RAG-only (no LLM synthesis)")
        print("\nAsk me anything about the lineage!")
        print("   Type 'help' for examples, 'quit' to exit")
        print("=" * 70 + "\n")

    def show_help(self):
        print("\nExample Queries:")
        print('   - What tables are in the catalog?')
        print('   - Show me lineage for [procedure_name]')
        print('   - Which procedures write to [table_name]?')
        print('   - What are the source tables for [table_name]?')
        print('   - How is [table_name].[column] computed?')
        print('   - What parameters filter [table_name]?')
        print('   - Show me high confidence attribute lineage')
        print('   - Find all procedures that use [table_name]')
        print('   - What columns does [table_name] have?')
        print('   - Explain the data flow from [table1] to [table2]')
        print()

    def _format_rag_only(self, rag_results: List[Dict[str, Any]]) -> str:
        if not rag_results:
            return '❌ No results found. Try rephrasing your query.'

        output = [f"✓ Found {len(rag_results)} results:\n"]
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in rag_results:
            rtype = r.get('type', 'unknown')
            by_type.setdefault(rtype, []).append(r)

        for rtype, items in by_type.items():
            output.append(rf"📌 {rtype.upper().replace('_', ' ')}:")
            for i, item in enumerate(items[:5], 1):
                score = item.get('score', item.get('score_stage1', 0.0))
                text = item['text']
                if len(text) > 150:
                    text = text[:150] + '...'
                output.append(f"   {i}. [score: {score:.2f}] {text}")
            if len(items) > 5:
                output.append(f"   ... and {len(items) - 5} more")
            output.append('')

        return '\n'.join(output)

    def _synthesize_with_llm(self, query: str, rag_results: List[Dict[str, Any]]) -> str:
        if not self.use_llm or not self.client:
            return self._format_rag_only(rag_results)

        context = '\n\n'.join(
            f"[{i+1}] {result['text']}" for i, result in enumerate(rag_results[:10])
        )

        prompt = (
            "You are a helpful data lineage assistant. Answer the user's question using the provided lineage information from the knowledge base.\n\n"
            "<lineage_knowledge_base>\n"
            f"{context}\n"
            "</lineage_knowledge_base>\n\n"
            f"User Question: {query}\n\n"
            "Instructions:\n"
            "1. Answer the question conversationally and accurately based on the lineage knowledge base\n"
            "2. Cite your sources using [1], [2], etc. to reference the numbered items from the knowledge base\n"
            "3. If the knowledge base doesn't contain enough information, say so\n"
            "4. Be concise but thorough\n"
            "5. Use technical terms appropriately (tables, procedures, columns, lineage)\n"
            "6. If the user asks for a flow or lineage, also include a Mermaid.js diagram syntax block that visualizes the flow as a graph, inside a ```mermaid code fence.\n\n"
            "Answer:"
        )

        try:
            message = self.client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=1000,
                messages=[{'role': 'user', 'content': prompt}],
            )
            answer = message.content[0].text

            references = '\n\n📚 References:\n'
            for i, result in enumerate(rag_results[:10], 1):
                refs_text = result['text'][:80].replace('\n', ' ')
                references += f"[{i}] {result['type']}: {refs_text}...\n"

            return answer + references
        except Exception as e:
            print(f'⚠️  LLM error: {e}')
            return self._format_rag_only(rag_results)

    def process_query(self, query: str) -> str:
        q_lower = query.lower()

        # Check for specific query types in order of specificity (most specific first)
        if 'hotspot' in q_lower or 'telemetry' in q_lower or 'imperva' in q_lower or 'call frequency' in q_lower:
            # Hotspot/telemetry queries -> search all types but prioritize telemetry docs
            doc_type = None
            k = 20
        elif 'parameter' in q_lower or '@' in q_lower:
            # Parameter queries (even if they mention procedure) -> search parameter_lineage
            doc_type = 'parameter_lineage'
            k = 15
        elif 'lineage' in q_lower or 'read from' in q_lower or 'write to' in q_lower or 'flow' in q_lower:
            # Lineage questions should search all doc types
            doc_type = None
            k = 15
        elif ('column' in q_lower or 'columns' in q_lower) and 'table' in q_lower:
            # Questions about columns IN a specific table -> search catalog
            doc_type = 'catalog'
            k = 15
        elif 'attribute' in q_lower or ('column' in q_lower and 'lineage' in q_lower):
            # Attribute/column lineage questions -> search attribute_lineage
            doc_type = 'attribute_lineage'
            k = 15
        elif 'table' in q_lower and 'catalog' in q_lower:
            doc_type = 'catalog'
            k = 20
        elif 'procedure' in q_lower and 'table' not in q_lower:
            # Only route to procedure if not asking about tables or parameters
            doc_type = 'procedure'
            k = 10
        elif 'compute' in q_lower or 'how' in q_lower:
            doc_type = None
            k = 15
        else:
            doc_type = None
            k = 10

        rag_results = self.rag.query(query, k=k, doc_type=doc_type)

        if self.use_llm and rag_results:
            body = self._synthesize_with_llm(query, rag_results)
        else:
            body = self._format_rag_only(rag_results)

        if 'lineage' in query.lower() or 'flow' in query.lower():
            body += '\n\n'

        conf = self.gt_validator.assess(query, body, rag_results)
        conf_line = self.gt_validator.format(conf)

        return body + '\n\n' + conf_line

    def run(self):
        self.welcome()
        while True:
            try:
                user_input = input('You: ').strip()
                if not user_input:
                    continue

                if user_input.lower() in ('quit', 'exit', 'q'):
                    print('\n👋 Goodbye!\n')
                    break

                if user_input.lower() == 'help':
                    self.show_help()
                    continue

                if user_input.lower() == 'stats':
                    print('\n📊 Statistics:')
                    for key, value in self.stats.items():
                        print(f'   • {key}: {value}')
                    print()
                    continue

                if user_input.lower() == 'history':
                    print('\n📜 Query History:')
                    for i, h in enumerate(self.history, 1):
                        print(f"   {i}. {h['query']}")
                    print()
                    continue

                print('\n🔍 Searching' + (' and synthesizing' if self.use_llm else '') + '...\n')
                answer = self.process_query(user_input)
                self.history.append({'query': user_input})
                print(answer)
                print()

            except KeyboardInterrupt:
                print('\n\n👋 Goodbye!\n')
                break
            except Exception as e:
                print(f'\n❌ Error: {e}\n')


def batch_query_mode(rag_path: Path, queries: List[str], use_llm: bool = True):
    interface = LineageChatInterface(rag_path, use_llm=use_llm)

    print('\n' + '=' * 70)
    print('BATCH QUERY MODE v5.0' + (' with LLM' if interface.use_llm else ' (RAG-only)'))
    print('=' * 70)

    for i, query in enumerate(queries, 1):
        print(f'\n📝 Query {i}: {query}')
        print('-' * 70)
        answer = interface.process_query(query)
        print(answer)


def main():
    # base_dir is parent of source_code (i.e., SPworkv4/)
    base_dir = Path(__file__).resolve().parent.parent
    rag_path = base_dir / 'output_artifacts' / 'lineage_rag.pkl'

    if not rag_path.exists():
        print(f'Error: RAG metadata not found at {rag_path}')
        print('   Please run the pipeline first:')
        print('     python parse_agent.py && python abstract_agent.py && python retrieve_agent.py')
        sys.exit(1)

    use_llm = '--no-llm' not in sys.argv
    if '--no-llm' in sys.argv:
        sys.argv.remove('--no-llm')

    if len(sys.argv) > 1:
        queries = sys.argv[1:]
        batch_query_mode(rag_path, queries, use_llm=use_llm)
    else:
        interface = LineageChatInterface(rag_path, use_llm=use_llm)
        interface.run()


if __name__ == "__main__":
    main()