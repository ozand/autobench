import json
import time
import sqlite3
import argparse
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runner import Runner

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
                
                cache_k = ctk if ctk != 'f16' else None
                cache_v = ctv if ctv != 'f16' else None
                
                res = Runner.run_local_vulkan(
                    prompt=test_prompt,
                    max_tokens=16,
                    device=dev,
                    model_path=model_path,
                    timeout=45,
                    context_length=ctx,
                    ts_split=ts,
                    split_mode=sm,
                    cache_type_k=cache_k,
                    cache_type_v=cache_v,
                    no_kv_offload=no_kv
                )
                
                status = res.get("status", "FAILED")
                prompt_ts = res.get("prompt_tokens_per_second") or 0.0
                eval_ts = res.get("eval_tokens_per_second") or 0.0
                out_text = res.get("raw_output", "")
                
                passed_needle = secret_needle in out_text
                retrieval = 1.0 if passed_needle else (0.5 if "8892" in out_text else 0.0)
                quality = 1.0 if status == "SUCCESS" else 0.0
                
                print(f"   -> Status: {status}, Prompt: {prompt_ts:.1f} t/s, Gen: {eval_ts:.1f} t/s, Retrieval: {retrieval*100:.0f}%", flush=True)
                
                cur.execute("""
                INSERT INTO model_benchmarks (
                    model_name, size_bytes, device, tensor_split, context_length,
                    kv_quant, kv_offload, prompt_tokens_per_sec, eval_tokens_per_sec,
                    retrieval_rate, quality_pass_rate, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, 0, dev, sm, ctx, ctk, 1 if no_kv else 0,
                    prompt_ts, eval_ts, retrieval, quality, status
                ))
                conn.commit()

    conn.close()
    print("Matrix execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()
    
    model_path = args.model_path
    if not model_path or "Program Files" in model_path:
        model_path = f"/home/opencode/llama.cpp/models/{args.model_name}"
        
    run_matrix(args.model_name, model_path)
