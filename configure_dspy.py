"""Quick script to configure DSPy with Ollama for use in the main agent.

Run this before running the agent to enable DSPy optimization.
Alternatively, set environment variables or configure in run_agent_hybrid.py
"""

import dspy


def configure():
    """Configure DSPy with Ollama model."""
    try:
        # Try to configure with Ollama
        lm = dspy.LM("ollama/phi3.5:3.8b-mini-instruct-q4_K_M")
        dspy.configure(lm=lm)
        print("✓ Successfully configured DSPy with Ollama!")
        print(f"  Model: phi3.5:3.8b-mini-instruct-q4_K_M")
        print("\nYou can now use DSPy modules in the agent.")
        return True
    except Exception as e:
        print(f"✗ Failed to configure Ollama: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is installed: https://ollama.com")
        print("2. Model is pulled: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")
        print("3. Ollama server is running: ollama serve")
        return False


if __name__ == "__main__":
    configure()
