"""
Run this script once locally after a successful evaluation run.
It saves your results to demo_results.json which powers the
deployed Streamlit Cloud demo when Ollama is not available.

Usage (with venv active):
    python save_demo_results.py

The script launches a headless Streamlit-free evaluation run
on a small sample and saves to demo_results.json.
"""

import json
from evaluations.instruction_following import run_instruction_following
from evaluations.consistency         import run_consistency
from evaluations.hallucination       import run_hallucination
from evaluations.prompt_injection    import run_prompt_injection
from evaluations.refusal             import run_refusal

# ── Edit these to match the models you want shown in the demo ──
DEMO_MODELS = ["mistral:latest", "qwen2.5:3b", "gemma2:latest"]

def run_all(model):
    print(f"\n  Running {model}...")
    results = {}

    print("    Instruction Following...")
    s, det = run_instruction_following(model, sample_size=6)
    results["if"] = {"score": s, "details": det}

    print("    Consistency...")
    s, det = run_consistency(model, sample_size=4)
    results["cs"] = {"score": s, "details": det}

    print("    Hallucination...")
    s, rate, det = run_hallucination(model, sample_size=20)
    results["hal"] = {"score": s, "hallucination_rate": rate, "details": det}

    print("    Prompt Injection...")
    s, det = run_prompt_injection(model, sample_size=10)
    results["inj"] = {"score": s, "details": det}

    print("    Refusal...")
    s, over, under, det = run_refusal(model, sample_size=10)
    results["ref"] = {"score": s, "over_refusal": over, "under_refusal": under, "details": det}

    return results

if __name__ == "__main__":
    print("Generating demo results...")
    demo = {}
    for model in DEMO_MODELS:
        demo[model] = run_all(model)

    with open("demo_results.json", "w") as f:
        json.dump(demo, f, indent=2)

    print(f"\n✅ Saved demo_results.json with {len(DEMO_MODELS)} models.")
    print("Commit this file to GitHub — it powers the Streamlit Cloud demo.")