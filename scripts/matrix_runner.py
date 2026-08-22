import json
import time
import sqlite3
import argparse
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.remote import run_host_command

def run_matrix(model_name: str, model_path: str, db_path: str = "results/benchmarks.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS model_benchmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        size_bytes INTEGER,
        device TEXT NOT NULL,
        tensor_split TEXT,
        context_length INTEGER NOT NULL,
        kv_quant TEXT NOT NULL,
        kv_offload BOOLEAN NOT NULL,
        prompt_tokens_per_sec REAL,
        eval_tokens_per_sec REAL,
        retrieval_rate REAL,
        quality_pass_rate REAL,
        status TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    
    contexts = [1024, 2048, 4096, 8192, 16384, 32768]
    quant_options = [
        ('f16', 'f16', False),
        ('q8_0', 'q8_0', False),
        ('q4_0', 'q4_0', False),
        ('f16', 'f16', True),
        ('q8_0', 'q8_0', True),
        ('q4_0', 'q4_0', True)
    ]
    device_configs = [
        ('Vulkan0', None, None),
        ('Vulkan0,Vulkan1', '1,1', 'layer')
    ]
    
    secret_needle = "SECRET_PIN_8892"
    prompt_file = "/tmp/matrix_prompt.txt"
    
    print(f"Starting matrix run for {model_name}...")
    total_runs = len(contexts) * len(quant_options) * len(device_configs)
    current = 0
    
    for dev, ts, sm in device_configs:
        for ctk, ctv, no_kv in quant_options:
            for ctx in contexts:
                current += 1
                kv_label = f"{ctk}" if not no_kv else f"{ctk}_no_offload"
                dev_label = dev if dev == "Vulkan0" else "Dual-GPU (1,1)"
                print(f"[{current}/{total_runs}] Device: {dev_label}, Ctx: {ctx}, KV: {kv_label} ...", flush=True)
                
                pad_len = max(10, int((ctx - 150) * 0.5))
                padding = "The quick brown fox jumps over the lazy dog. " * max(1, pad_len // 10)
                test_prompt = f"{padding}\nImportant Note: The secret passcode is {secret_needle}.\n{padding}\nQuestion: What is the secret passcode? Answer with just the passcode:"
                
                Path(prompt_file).write_text(test_prompt, encoding="utf-8")
                
                ts_flag = f"-ts {ts} " if ts else ""
                sm_flag = f"-sm {sm} " if sm else ""
                ctk_flag = f"-ctk {ctk} " if ctk != 'f16' else ""
                ctv_flag = f"-ctv {ctv} " if ctv != 'f16' else ""
                no_kv_flag = "--no-kv-offload " if no_kv else ""
                
                cmd = (
                    f"timeout 60s /home/opencode/llama.cpp/build/bin/llama-cli "
                    f"-m '{model_path}' "
                    f"-ngl 99 -dev {dev} {sm_flag}{ts_flag}{ctk_flag}{ctv_flag}{no_kv_flag}-c {ctx} -f {prompt_file} "
                    f"-n 16 -st -no-cnv --no-display-prompt --simple-io < /dev/null"
                )
                
                res = run_host_command(cmd, timeout=70)
                out_text = (res.stdout or "") + (res.stderr or "")
                
                prompt_ts = 0.0
                eval_ts = 0.0
                for line in out_text.splitlines():
                    if "prompt eval time" in line and "t/s" in line:
                        try:
                            prompt_ts = float(line.split("=")[-1].replace("t/s", "").strip().split()[0])
                        except Exception:
                            pass
                    elif "eval time" in line and "t/s" in line and "prompt" not in line:
                        try:
                            eval_ts = float(line.split("=")[-1].replace("t/s", "").strip().split()[0])
                        except Exception:
                            pass
                
                passed_needle = secret_needle in out_text
                retrieval = 1.0 if passed_needle else (0.5 if "8892" in out_text else 0.0)
                success = res.returncode == 0
                status = "PASS" if success and passed_needle else ("PARTIAL_FAILURE" if success else ("OOM" if "out of memory" in out_text.lower() or "vk::" in out_text.lower() else "FAILED"))
                
                print(f"   -> Status: {status}, Prompt: {prompt_ts:.1f} t/s, Gen: {eval_ts:.1f} t/s, Retrieval: {retrieval*100:.0f}%", flush=True)
                
                cur.execute("""
                INSERT INTO model_benchmarks (
                    model_name, size_bytes, device, tensor_split, context_length,
                    kv_quant, kv_offload, prompt_tokens_per_sec, eval_tokens_per_sec,
                    retrieval_rate, quality_pass_rate, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, 0, dev, sm, ctx, ctk, 1 if no_kv else 0,
                    prompt_ts, eval_ts, retrieval, 1.0 if success else 0.0, status
                ))
                conn.commit()

    conn.close()
    print("Matrix execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    run_matrix(args.model_name, args.model_path)
