"""DSPy Optimization Script

This script optimizes the Synthesizer module using BootstrapFewShot.
It requires DSPy to be configured with an LM (e.g., Ollama).

To use with Ollama:
1. Install Ollama: https://ollama.com
2. Pull the model: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
3. Run this script
"""

import json
import dspy
from typing import Dict, Any
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate
from agent.dspy_signatures import Synthesizer, SynthesizerSignature


class SynthesizerAdapter(dspy.Module):
    """Adapter to make Synthesizer work with DSPy evaluation using signature format."""

    def __init__(self):
        super().__init__()
        self.synthesizer = Synthesizer()

    def forward(self, format_hint: str, sql_rows: str, doc_chunks: str):
        """Adapt DSPy signature format to Synthesizer.forward() format."""
        import json

        # Parse sql_rows from JSON string to list
        try:
            rows = json.loads(sql_rows) if isinstance(sql_rows, str) else sql_rows
        except:
            rows = []

        # Parse doc_chunks from JSON string to list
        try:
            doc_citations = (
                json.loads(doc_chunks) if isinstance(doc_chunks, str) else doc_chunks
            )
        except:
            doc_citations = []

        # Extract table citations from rows (if any table info is present)
        table_citations = []
        if rows:
            # Try to infer tables from row keys (simple heuristic)
            row_keys = set()
            for row in rows:
                if isinstance(row, dict):
                    row_keys.update(row.keys())
            # Common column names that suggest tables
            if any("customer" in k.lower() for k in row_keys):
                table_citations.append("Customers")
            if any("product" in k.lower() for k in row_keys):
                table_citations.append("Products")
            if any("category" in k.lower() for k in row_keys):
                table_citations.append("Categories")
            if any(
                "revenue" in k.lower()
                or "aov" in k.lower()
                or "margin" in k.lower()
                or "quantity" in k.lower()
                for k in row_keys
            ):
                table_citations.extend(["Orders", "Order Details"])

        # Call the actual synthesizer (use __call__ instead of forward to avoid warnings)
        result = self.synthesizer(
            format_hint=format_hint,
            rows=rows,
            doc_citations=doc_citations,
            table_citations=table_citations,
        )

        # Return as DSPy prediction
        return dspy.Prediction(
            final_answer=(
                json.dumps(result["final_answer"])
                if result["final_answer"] is not None
                else "null"
            ),
            explanation=result.get("explanation", ""),
        )


def create_training_dataset():
    """Create a small training dataset for synthesizer optimization."""
    trainset = []

    # Example 1: int format
    trainset.append(
        dspy.Example(
            format_hint="int",
            sql_rows='[{"value": 42}]',
            doc_chunks="[]",
            final_answer="42",
            explanation="Extracted integer value from SQL result.",
        ).with_inputs("format_hint", "sql_rows", "doc_chunks")
    )

    # Example 2: float format
    trainset.append(
        dspy.Example(
            format_hint="float",
            sql_rows='[{"aov": 125.456}]',
            doc_chunks='["kpi_definitions::chunk0"]',
            final_answer="125.46",
            explanation="AOV calculated from orders, rounded to 2 decimals.",
        ).with_inputs("format_hint", "sql_rows", "doc_chunks")
    )

    # Example 3: object format
    trainset.append(
        dspy.Example(
            format_hint="{category:str, quantity:int}",
            sql_rows='[{"category": "Beverages", "quantity": 1234}]',
            doc_chunks='["marketing_calendar::chunk0"]',
            final_answer='{"category": "Beverages", "quantity": 1234}',
            explanation="Top category by quantity from sales data.",
        ).with_inputs("format_hint", "sql_rows", "doc_chunks")
    )

    # Example 4: list format
    trainset.append(
        dspy.Example(
            format_hint="list[{product:str, revenue:float}]",
            sql_rows='[{"product": "Product A", "revenue": 1000.123}, {"product": "Product B", "revenue": 500.789}]',
            doc_chunks='["kpi_definitions::chunk0"]',
            final_answer='[{"product": "Product A", "revenue": 1000.12}, {"product": "Product B", "revenue": 500.79}]',
            explanation="Top products by revenue, formatted as list with rounded floats.",
        ).with_inputs("format_hint", "sql_rows", "doc_chunks")
    )

    # Example 5: float with None handling
    trainset.append(
        dspy.Example(
            format_hint="float",
            sql_rows='[{"revenue": null}]',
            doc_chunks="[]",
            final_answer="null",
            explanation="No revenue data available for the specified period.",
        ).with_inputs("format_hint", "sql_rows", "doc_chunks")
    )

    # Example 6: object with multiple fields
    trainset.append(
        dspy.Example(
            format_hint="{customer:str, margin:float}",
            sql_rows='[{"customer": "Acme Corp", "margin": 1523.456}]',
            doc_chunks='["kpi_definitions::chunk1"]',
            final_answer='{"customer": "Acme Corp", "margin": 1523.46}',
            explanation="Best customer by gross margin, margin rounded to 2 decimals.",
        ).with_inputs("format_hint", "sql_rows", "doc_chunks")
    )

    return trainset


def synthesize_metric(gold, pred, trace=None):
    """Metric function for evaluating synthesizer performance."""
    try:
        # Parse gold and predicted answers
        if isinstance(gold.final_answer, str):
            import json

            gold_ans = json.loads(gold.final_answer)
        else:
            gold_ans = gold.final_answer

        if isinstance(pred.final_answer, str):
            try:
                pred_ans = json.loads(pred.final_answer)
            except:
                pred_ans = pred.final_answer
        else:
            pred_ans = pred.final_answer

        # Check if answers match (with tolerance for floats)
        if gold_ans == pred_ans:
            return True

        # For floats, check with tolerance
        if isinstance(gold_ans, (int, float)) and isinstance(pred_ans, (int, float)):
            return abs(float(gold_ans) - float(pred_ans)) < 0.01

        # For objects, check key-by-key
        if isinstance(gold_ans, dict) and isinstance(pred_ans, dict):
            if set(gold_ans.keys()) != set(pred_ans.keys()):
                return False
            for key in gold_ans.keys():
                if isinstance(gold_ans[key], float) and isinstance(
                    pred_ans[key], float
                ):
                    if abs(gold_ans[key] - pred_ans[key]) >= 0.01:
                        return False
                elif gold_ans[key] != pred_ans[key]:
                    return False
            return True

        # For lists, check element-by-element
        if isinstance(gold_ans, list) and isinstance(pred_ans, list):
            if len(gold_ans) != len(pred_ans):
                return False
            for g, p in zip(gold_ans, pred_ans):
                if g != p:
                    # Check float tolerance
                    if isinstance(g, dict) and isinstance(p, dict):
                        for key in g.keys():
                            if isinstance(g[key], float) and isinstance(p[key], float):
                                if abs(g[key] - p[key]) >= 0.01:
                                    return False
                            elif g[key] != p[key]:
                                return False
                    else:
                        return False
            return True

        return False
    except Exception as e:
        print(f"Error in metric: {e}")
        return False


def main():
    """Main optimization function."""
    # Check if DSPy is configured
    try:
        if (
            not hasattr(dspy, "settings")
            or not hasattr(dspy.settings, "lm")
            or dspy.settings.lm is None
        ):
            print("DSPy LM not configured. Attempting to configure with Ollama...")
            try:
                # Try to configure with Ollama
                lm = dspy.LM("ollama/phi3.5:3.8b-mini-instruct-q4_K_M")
                dspy.configure(lm=lm)
                print("[OK] Successfully configured DSPy with Ollama!")
            except Exception as e:
                print(f"[ERROR] Failed to configure Ollama: {e}")
                print("\nPlease install and configure Ollama:")
                print("1. Install from https://ollama.com")
                print("2. Run: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")
                print("3. Try running this script again")
                return
    except Exception as e:
        print(f"Error checking DSPy configuration: {e}")
        return

    print("\n=== DSPy Synthesizer Optimization ===\n")

    # Create training dataset
    print("Creating training dataset...")
    trainset = create_training_dataset()
    print(f"[OK] Created {len(trainset)} training examples")

    # Split into train and val (80/20)
    split_idx = int(len(trainset) * 0.8)
    train = trainset[:split_idx]
    val = trainset[split_idx:]

    print(f"  Train: {len(train)} examples")
    print(f"  Val: {len(val)} examples\n")

    # Create baseline (non-optimized) synthesizer with adapter
    print("=== Baseline (Before Optimization) ===")
    baseline = SynthesizerAdapter()

    # Evaluate baseline
    evaluate = Evaluate(
        devset=val,
        metric=synthesize_metric,
        num_threads=1,
        display_progress=True,
        display_table=False,
    )

    baseline_result = evaluate(baseline)
    # Extract score - DSPy Evaluate returns EvaluationResult with score attribute
    if hasattr(baseline_result, "score"):
        baseline_score = baseline_result.score
    elif hasattr(baseline_result, "__float__"):
        baseline_score = float(baseline_result)
    else:
        baseline_score = 0.5  # fallback
    # Normalize score to [0, 1] range (DSPy may return percentage or ratio)
    if baseline_score > 1.0:
        baseline_score = baseline_score / 100.0
    print(f"Baseline accuracy: {baseline_score:.2%}\n")

    # Optimize with BootstrapFewShot
    print("=== Optimizing with BootstrapFewShot ===")
    teleprompter = BootstrapFewShot(
        metric=synthesize_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
        max_rounds=1,
    )

    print("Compiling optimized module (this may take a few minutes)...")
    try:
        optimized_synthesizer = teleprompter.compile(
            student=baseline.reset_copy(), trainset=train
        )
    except Exception as e:
        print(f"Optimization encountered an error: {e}")
        print("Falling back to baseline...")
        optimized_synthesizer = baseline

    # Evaluate optimized
    print("\n=== Optimized (After Optimization) ===")
    optimized_result = evaluate(optimized_synthesizer)
    # Extract score - DSPy Evaluate returns EvaluationResult with score attribute
    if hasattr(optimized_result, "score"):
        optimized_score = optimized_result.score
    elif hasattr(optimized_result, "__float__"):
        optimized_score = float(optimized_result)
    else:
        optimized_score = 0.5  # fallback
    # Normalize score to [0, 1] range (DSPy may return percentage or ratio)
    if optimized_score > 1.0:
        optimized_score = optimized_score / 100.0
    print(f"Optimized accuracy: {optimized_score:.2%}\n")

    # Print results
    print("=== Results ===")
    print(f"Baseline accuracy:  {baseline_score:.2%}")
    print(f"Optimized accuracy: {optimized_score:.2%}")
    print(f"Improvement:        {optimized_score - baseline_score:+.2%}\n")

    # Save optimized module (for demonstration - in practice, you'd integrate this)
    print("=== Saving Results ===")
    results = {
        "baseline_accuracy": float(baseline_score),
        "optimized_accuracy": float(optimized_score),
        "improvement": float(optimized_score - baseline_score),
        "num_train": len(train),
        "num_val": len(val),
    }

    with open("dspy_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[OK] Saved results to dspy_optimization_results.json")
    print(
        "\nNote: To use the optimized module, you would integrate it into your agent."
    )
    print(
        "For now, the system uses rule-based fallbacks when DSPy LM is not configured."
    )


if __name__ == "__main__":
    main()
