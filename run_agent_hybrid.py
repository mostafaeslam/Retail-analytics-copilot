"""LangGraph-like skeleton (minimal, runnable without real LangGraph dependency)

This file implements a stepwise hybrid pipeline with the required nodes
(router, retriever, planner, nl->sql, executor, synthesizer, repair loop).
"""

import argparse
import json
import os

# Optional: Configure DSPy with Ollama if available
# Uncomment the following lines to enable DSPy optimization:
# try:
#     import dspy
#     dspy.configure(lm=dspy.LM('ollama/phi3.5:3.8b-mini-instruct-q4_K_M'))
#     print("DSPy configured with Ollama")
# except:
#     print("DSPy not configured, using rule-based fallbacks")

from agent.graph_hybrid import HybridAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch", required=True, help="Input JSONL file with questions"
    )
    parser.add_argument("--out", required=True, help="Output JSONL file for answers")
    args = parser.parse_args()

    # Initialize the hybrid agent
    agent = HybridAgent(db_path="data/northwind.sqlite", docs_dir="docs")

    outputs = []
    with open(args.batch, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            question_obj = json.loads(line)
            out = agent.run(question_obj)
            outputs.append(out)

    # Write outputs to JSONL
    with open(args.out, "w", encoding="utf-8") as fo:
        for o in outputs:
            fo.write(json.dumps(o) + "\n")

    print(f"Wrote {len(outputs)} outputs to {args.out}")


if __name__ == "__main__":
    main()
