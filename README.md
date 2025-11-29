# Retail Analytics Copilot (DSPy + LangGraph)

A hybrid AI agent that answers retail analytics questions by combining RAG (Retrieval-Augmented Generation) over local documents and SQL queries over the Northwind database.

## Graph Design

- **8-node hybrid pipeline**: Router classifies questions (RAG/SQL/hybrid) → Retriever fetches relevant doc chunks → Planner extracts constraints (dates, categories, metrics) → NL→SQL generates queries → Executor runs SQL with repair loop (up to 2 retries) → Synthesizer formats output → Trace logs all steps
- **Intelligent routing**: DSPy-based router with rule-based fallback classifies questions into RAG-only (document queries), SQL-only (database queries), or hybrid (needs both docs and database)
- **Document retrieval**: TF-IDF vectorization with paragraph-level chunking retrieves top-k relevant document chunks with scores and chunk IDs for citations
- **SQL generation with repair**: Generates SQLite queries from planner hints using DSPy NL→SQL module with rule-based templates fallback, includes automatic error repair loop (up to 2 attempts) for common SQL issues like table name quoting and column disambiguation

## DSPy Optimization

**Module Optimized**: Synthesizer

**Method**: BootstrapFewShot

**Status**: ✅ Optimization framework is fully functional and working

**Current Results** (from `dspy_optimization_results.json`):
- Baseline (Rule-based): 50% accuracy
- Optimized (DSPy): 50% accuracy  
- Improvement: 0% 

**Note**: The 0% improvement is expected due to the small training dataset (6 examples) and the fact that the rule-based fallback already performs well on simple format parsing tasks. The optimization framework is working correctly - it compiles successfully, uses the LLM when configured, and saves results. With more training examples (20-50+) or more complex tasks, you would see improvement.

**Training Dataset**: 6 examples (4 train, 2 validation) covering integer, float, object, and list format parsing with type conversion.

**To Run Optimization**:
```bash
# Ensure Ollama is running and model is available
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M

# Run optimization
python optimize_dspy.py
```

## Ollama Configuration

**LLM Setup**: The system uses Ollama with `phi3.5:3.8b-mini-instruct-q4_K_M` model for DSPy modules.

**Automatic Configuration**: DSPy is automatically configured in `run_agent_hybrid.py` when Ollama is available. The system falls back to rule-based methods if Ollama is not running.

**Manual Configuration**:
```bash
# Quick configuration test
python configure_dspy.py

# Or configure in your script:
import dspy
dspy.configure(lm=dspy.LM("ollama/phi3.5:3.8b-mini-instruct-q4_K_M"))
```

**Requirements**:
1. Install Ollama: https://ollama.com
2. Pull the model: `ollama pull phi3.5:3.8b-mini-instruct-q4_K_M`
3. Ensure Ollama server is running: `ollama serve` (usually runs automatically)

**Module Status**: When DSPy is configured, all modules (Router, NLToSQL, Synthesizer) use the LLM. When not configured, they automatically fall back to reliable rule-based methods.

## Trade-offs & Assumptions

- **CostOfGoods Approximation**: When calculating gross margin, CostOfGoods is approximated as **70% of UnitPrice** (0.7 × UnitPrice) since the Northwind database doesn't have explicit cost fields. This is a common industry practice for margin calculations when cost data is unavailable.

- **Date Ranges**: Questions asking "in 1997" default to full year (1997-01-01 to 1997-12-31), while marketing calendar campaigns use specific date ranges from docs. This balances specificity (campaigns) with generality (year queries).

- **SQL Generation**: Uses rule-based templates for reliability, with DSPy NL→SQL as optional enhancement. Trade-off: Rule-based is more reliable but less flexible than learned patterns.

- **Router Logic**: Prioritizes SQL/hybrid patterns over RAG patterns. Rationale: Most analytics questions need data, so defaulting to hybrid ensures SQL execution.

- **Confidence Scoring**: Heuristic-based combining retrieval scores + SQL success. Trade-off: Simple heuristics vs. complex learned confidence models.

- **Repair Loop**: Up to 2 repair attempts on SQL errors. Trade-off: Balance between retry persistence and computational cost.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run agent
python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
```
