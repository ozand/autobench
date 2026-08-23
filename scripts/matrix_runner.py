import json
import time
import sqlite3
import argparse
import sys
import os
import re
import shlex
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.remote import run_host_command, host_command

def parse_output(stdout: str, stderr: str, returncode: int):
    diag = f"{stdout}\n{stderr}".lower()
    if returncode != 0:
        if "out of device memory" in diag or "erroroutofdevicememory" in diag or "failed to allocate" in diag:
            return {"status": "OOM", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": ""}
        if "does not support split buffers" in diag:
            return {"status": "UNSUPPORTED_BACKEND", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": ""}
        return {"status": "FAILED", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": stdout}
    
    speed_match = re.search(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", stdout)
    prompt_ts = float(speed_match.group(1)) if speed_match else 0.0
    gen_ts = float(speed_match.group(2)) if speed_match else 0.0
    
    return {
        "status": "SUCCESS",
        "prompt_ts": prompt_ts,
        "gen_ts": gen_ts,
        "raw_output": stdout
    }

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
        ('Vulkan1', None, None),
        ('Vulkan0,Vulkan1', '1,1', 'layer')
    ]
    
    secret_needle = "SECRET_PIN_8892"
    print(f"Starting matrix run for {model_name}...")
    total_runs = len(contexts) * len(quant_options) * len(device_configs)
    current = 0
    
    for dev, ts, sm in device_configs:
        for ctk, ctv, no_kv in quant_options:
            failed_state = None
            for ctx in contexts:
                current += 1
                kv_label = f"{ctk}" if not no_kv else f"{ctk}_no_offload"
                dev_label = dev if "," not in dev else "Dual-GPU (1,1)"
                if failed_state is not None and "," not in dev:
                    print(f"[{current}/{total_runs}] Device: {dev_label}, Ctx: {ctx}, KV: {kv_label} ... (cascaded {failed_state})", flush=True)
                    cur.execute("DELETE FROM model_benchmarks WHERE model_name=? AND device=? AND context_length=? AND kv_quant=? AND kv_offload=?", (model_name, dev, ctx, ctk, 1 if no_kv else 0))
                    cur.execute("INSERT INTO model_benchmarks (model_name, size_bytes, device, tensor_split, context_length, kv_quant, kv_offload, prompt_tokens_per_sec, eval_tokens_per_sec, retrieval_rate, quality_pass_rate, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (model_name, 0, dev, sm, ctx, ctk, 1 if no_kv else 0, 0.0, 0.0, 0.0, 0.0, failed_state))
                    conn.commit()
                    continue
                print(f"[{current}/{total_runs}] Device: {dev_label}, Ctx: {ctx}, KV: {kv_label} ...", flush=True)
                
                cur.execute("""
                    SELECT prompt_tokens_per_sec, eval_tokens_per_sec, status FROM model_benchmarks 
                    WHERE model_name=? AND device=? AND context_length=? AND kv_quant=? AND kv_offload=?
                """, (model_name, dev, ctx, ctk, 1 if no_kv else 0))
                row = cur.fetchone()
                if row and row[0] > 0 and row[2] == "SUCCESS":
                    print(f"   -> (Already present: {row[0]:.1f} t/s, Gen: {row[1]:.1f} t/s, status: {row[2]})", flush=True)
                    continue
                
                # Cleanup any stale processes before test
                run_host_command("killall -9 llama-cli")
                
                pad_words = max(5, int((ctx - 120) * 0.35))
                gen_py = (
                    f"python3 -c \\\""
                    f"pad = 'The quick brown fox jumps over the lazy dog. ' * {max(1, pad_words // 9)}; "
                    f"p = f'{{pad}}\nImportant Note: The secret passcode is {secret_needle}.\n{{pad}}\nQuestion: What is the secret passcode? Answer with just the passcode:'; "
                    f"open('/tmp/autobench_prompt.txt', 'w').write(p)\\\""
                )
                run_host_command(gen_py)
                
                cache_k = ctk if ctk != 'f16' else None
                cache_v = ctv if ctv != 'f16' else None

                q_model_path = shlex.quote(model_path)
                q_dev = shlex.quote(dev)
                ts_flag = f"-ts {shlex.quote(ts)} " if ts else ("-ts 1,1 " if "," in dev else "")
                sm_flag = f"-sm {shlex.quote(sm)} " if sm else ""
                ctk_flag = f"-ctk {shlex.quote(cache_k)} " if cache_k else ""
                ctv_flag = f"-ctv {shlex.quote(cache_v)} " if cache_v else ""
                no_kv_flag = "--no-kv-offload " if no_kv else ""

                cmd = (
                    f"/home/opencode/llama.cpp/build/bin/llama-cli "
                    f"-m {q_model_path} -ngl 99 -dev {q_dev} {sm_flag}{ts_flag}{ctk_flag}{ctv_flag}{no_kv_flag}"
                    f"-c {ctx} -f /tmp/autobench_prompt.txt -n 16 -st -no-cnv --no-display-prompt --simple-io < /dev/null"
                )
                
                # Optimized prompt & timeout
                timeout_sec = 10 if ctx <= 2048 else (18 if ctx <= 8192 else 25)
                
                start_t = time.time()
                try:
                    proc = run_host_command(cmd, timeout=timeout_sec)
                    res = parse_output(proc.stdout, proc.stderr, proc.returncode)
                except subprocess.TimeoutExpired:
                    run_host_command("killall -9 llama-cli")
                    res = {"status": "TIMEOUT", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": ""}
                
                status = res["status"]
                prompt_ts = res["prompt_ts"]
                eval_ts = res["gen_ts"]
                out_text = res["raw_output"]
                
                passed_needle = secret_needle in out_text
                retrieval = 1.0 if passed_needle else (0.5 if "8892" in out_text else 0.0)
                quality = 1.0 if status == "SUCCESS" else 0.0
                
                if status in ("TIMEOUT", "OOM", "FAILED") and "," not in dev:
                    failed_state = status
                print(f"   -> Status: {status}, Prompt: {prompt_ts:.1f} t/s, Gen: {eval_ts:.1f} t/s, Retrieval: {retrieval*100:.0f}%", flush=True)
                
                cur.execute("""
                DELETE FROM model_benchmarks 
                WHERE model_name=? AND device=? AND context_length=? AND kv_quant=? AND kv_offload=?
                """, (model_name, dev, ctx, ctk, 1 if no_kv else 0))
                
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
    
    # Export to JSON
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_benchmarks ORDER BY model_name, context_length")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    
    json_path = Path("docs/benchmarks_data.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(rows)} rows to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()
    
    model_path = args.model_path
    if not model_path or "Program Files" in model_path:
        model_path = f"/home/opencode/llama.cpp/models/{args.model_name}"
        
    run_matrix(args.model_name, model_path)