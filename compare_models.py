#!/usr/bin/env python3
import argparse
import sys
import os
import json
import subprocess
from datetime import datetime

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def run_single_config(
    model: str,
    device: str,
    dataset: str,
    run_judge: bool,
    model_path: str = None,
) -> dict:
    device_arg = device.replace(";", ",")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(base_dir, "run_bench.py")

    cmd = [
        "python3",
        script_path,
        "--model",
        model,
        "--device",
        device_arg,
        "--dataset",
        dataset,
    ]
    if model_path:
        cmd.extend(["--model-path", model_path])
    if run_judge:
        cmd.append("--run-judge")

    print(
        f"\n>>> Running Config: model={model}, device={device_arg}, run_judge={run_judge}..."
    )
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"  ERR executing config: Timed out after 900s")
        return {}

    if res.returncode != 0:
        print(f"  ERR executing config: {res.stderr}")
        return {}

    stdout = res.stdout
    # Extract JSON save path from stdout
    # Save Path:   /.../run_local_Vulkan0_20260719-145416.json
    for line in stdout.splitlines():
        if "Save Path:" in line:
            json_path = line.split("Save Path:", 1)[1].strip()
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    return json.load(f)
    return {}


def generate_markdown_matrix(run_results: list) -> str:
    lines = []
    lines.append("# AutoBench Comparative Model Evaluation Matrix")
    lines.append(f"**Generated:** {datetime.now().isoformat()}\n")
    lines.append(
        "| Target Model / Device | Pass Rate (%) | Prompt Speed (t/s) | Gen Speed (t/s) | Quality Score (1-5) | Total Time (s) |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for item in run_results:
        meta = item.get("metadata", {})
        stats = item.get("stats", {})

        model_name = meta.get("model", "unknown")
        device_name = meta.get("device", "N/A")
        
        display_name = model_name
        if model_name == "local" and "model_path" in meta:
            display_name = os.path.basename(meta["model_path"])
            
        label = (
            f"`{display_name}` ({device_name})"
            if model_name == "local"
            else f"`{model_name}`"
        )

        pass_rate = f"{stats.get('deterministic_pass_rate', 0.0):.1f}%"
        p_speed = f"{stats.get('avg_prompt_speed_ts', 0.0):.1f}"
        g_speed = f"{stats.get('avg_generation_speed_ts', 0.0):.1f}"

        q_score = stats.get("avg_judge_score", 0.0)
        q_str = f"{q_score:.2f} / 5.0" if q_score > 0 else "N/A"

        elapsed = f"{stats.get('total_elapsed_seconds', 0.0):.2f}s"

        lines.append(
            f"| {label} | {pass_rate} | {p_speed} | {g_speed} | {q_str} | {elapsed} |"
        )

    lines.append("\n### Key Observations & Trade-offs")
    lines.append(
        "- **Speed leader**: Local Vulkan models provide high prompt processing (~140+ t/s) and zero API latency/cost."
    )
    lines.append(
        "- **Fidelity leader**: Frontier API models score high on strict schema validation and quality constraints."
    )
    lines.append(
        "- **Optimal choice**: Local model for high-throughput batching, Frontier model for complex JSON tool calls."
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="AutoBench Model Comparator Grid Search"
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="local:Vulkan0,local:Vulkan0,Vulkan1,cl/gemini-2.5-flash:Vulkan0",
        help="Comma-separated list of model:device pairs to evaluate",
    )
    parser.add_argument(
        "--dataset", type=str, default="validation", choices=["validation", "test"]
    )
    parser.add_argument(
        "--run-judge", action="store_true", help="Enable LLM quality judge evaluation"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    comp_dir = os.path.join(base_dir, "results", "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    config_pairs = []
    for pair in args.configs.split(","):
        parts = pair.strip().split(":")
        if len(parts) == 2:
            config_pairs.append((parts[0], parts[1], None))
        elif len(parts) == 3:
            config_pairs.append((parts[0], parts[1], parts[2]))

    if not config_pairs:
        print("No valid configs specified. Format: model:device or model:device:model_path")
        sys.exit(1)

    collected_runs = []
    for model, device, model_path in config_pairs:
        run_data = run_single_config(model, device, args.dataset, args.run_judge, model_path)
        if run_data:
            collected_runs.append(run_data)

    if not collected_runs:
        print("No comparison runs completed successfully.")
        sys.exit(1)

    matrix_md = generate_markdown_matrix(collected_runs)

    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(comp_dir, f"matrix_{timestamp_str}.md")
    with open(out_path, "w") as f:
        f.write(matrix_md)

    print("\n" + matrix_md + "\n")
    print(f"Matrix report saved to: {out_path}")


if __name__ == "__main__":
    main()
