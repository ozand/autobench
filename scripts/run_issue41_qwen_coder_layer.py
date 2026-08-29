#!/usr/bin/env python3
"""Run the one reviewed Issue 41 dual-GPU layer follow-up."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from authoritative_bench import build_plan, discover_remote_models, execute_suite

MODEL_NAME = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
OUTPUT = Path("results/issue41-followup-qwen-coder-layer.json")

models = [model for model in discover_remote_models() if model["name"] == MODEL_NAME]
if len(models) != 1:
    raise RuntimeError("exact reviewed model was not uniquely discovered")

plan = build_plan(models, "suite")
plan["models"][0]["configurations"] = [
    {
        "device": "Vulkan0,Vulkan1",
        "tensor_split": "1,1",
        "split_mode": "layer",
        "mode": "full",
    }
]
result = execute_suite(
    plan,
    timeout=180,
    context_sizes=[1024, 2048, 4096, 8192],
    boundary_step=256,
    retrieval_repetitions=3,
    reliability_threshold=0.0,
    performance_context=1024,
    prompt_tokens=512,
    output_tokens=64,
    warmups=1,
    performance_repetitions=2,
    dataset_dir="datasets/validation",
    max_tasks=2,
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
row = result["models"][0]["configurations"][0].get("result", {})
print(json.dumps({"status": row.get("status"), "execution_status": result["execution_status"], "authoritative": result["authoritative"]}))
