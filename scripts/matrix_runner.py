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


def classify_row(status: str, prompt_ts, gen_ts) -> str:
    """Classify whether a row is safe for authoritative publication."""
    speed_ok = (
        isinstance(prompt_ts, (int, float))
        and isinstance(gen_ts, (int, float))
        and prompt_ts > 0
        and gen_ts > 0
    )
    if status == "SUCCESS" and speed_ok:
        return "AUTHORITATIVE"
    if status == "SUCCESS":
        return "AMBIGUOUS"
    return "NON_AUTHORITATIVE"


def ensure_schema(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(model_benchmarks)")}
    if "publication_class" not in columns:
        conn.execute(
            "ALTER TABLE model_benchmarks ADD COLUMN publication_class TEXT NOT NULL DEFAULT 'AMBIGUOUS'"
        )
    if "provenance" not in columns:
        conn.execute(
            "ALTER TABLE model_benchmarks ADD COLUMN provenance TEXT NOT NULL DEFAULT 'legacy_audit'"
        )


def audit_existing_rows(conn) -> dict[str, int]:
    """Classify legacy rows without deleting recoverable historical evidence."""
    ensure_schema(conn)
    rows = conn.execute(
        "SELECT id, status, prompt_tokens_per_sec, eval_tokens_per_sec FROM model_benchmarks"
    ).fetchall()
    counts = {"AUTHORITATIVE": 0, "AMBIGUOUS": 0, "NON_AUTHORITATIVE": 0}
    for row_id, status, prompt_ts, gen_ts in rows:
        publication_class = classify_row(status, prompt_ts, gen_ts)
        reason = (
            "measured_success_with_positive_speed"
            if publication_class == "AUTHORITATIVE"
            else "success_without_complete_speed_metrics"
            if publication_class == "AMBIGUOUS"
            else "execution_not_success"
        )
        conn.execute(
            "UPDATE model_benchmarks SET publication_class=?, provenance=? WHERE id=?",
            (publication_class, f"legacy_audit:{reason}", row_id),
        )
        counts[publication_class] += 1
    conn.commit()
    return counts


def export_dashboard(conn, json_path: str = "docs/benchmarks_data.json"):
    conn.row_factory = sqlite3.Row
    rows = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM model_benchmarks ORDER BY model_name, context_length, id"
        )
    ]
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def parse_output(stdout: str, stderr: str, returncode: int):
    diag = f"{stdout}\n{stderr}".lower()
    if returncode != 0:
        if "out of device memory" in diag or "erroroutofdevicememory" in diag or "failed to allocate" in diag:
            return {"status": "OOM", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": ""}
        if "does not support split buffers" in diag:
            return {"status": "UNSUPPORTED_BACKEND", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": ""}
        return {"status": "FAILED", "prompt_ts": 0.0, "gen_ts": 0.0, "raw_output": stdout}
    
    combined = f"{stdout}\n{stderr}"
    speed_match = re.search(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", combined)
    if not speed_match:
        speed_match = re.search(r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s", combined)
    prompt_ts = float(speed_match.group(1)) if speed_match else 0.0
    gen_ts = float(speed_match.group(2)) if speed_match else 0.0
    
    return {
        "status": "SUCCESS" if speed_match else "METRIC_PARSE_FAILED",
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
        publication_class TEXT NOT NULL DEFAULT 'AMBIGUOUS',
        provenance TEXT NOT NULL DEFAULT 'matrix_run',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    ensure_schema(conn)
    audit_existing_rows(conn)
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
                if failed_state is not None :
                    print(f"[{current}/{total_runs}] Device: {dev_label}, Ctx: {ctx}, KV: {kv_label} ... (cascaded {failed_state})", flush=True)
                    cur.execute("DELETE FROM model_benchmarks WHERE model_name=? AND device=? AND context_length=? AND kv_quant=? AND kv_offload=?", (model_name, dev, ctx, ctk, 1 if no_kv else 0))
                    cur.execute("INSERT INTO model_benchmarks (model_name, size_bytes, device, tensor_split, context_length, kv_quant, kv_offload, prompt_tokens_per_sec, eval_tokens_per_sec, retrieval_rate, quality_pass_rate, status, publication_class, provenance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (model_name, 0, dev, sm, ctx, ctk, 1 if no_kv else 0, None, None, None, None, failed_state, "NON_AUTHORITATIVE", "matrix_run:cascaded_failure"))
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
                publication_class = classify_row(status, prompt_ts, eval_ts)
                out_text = res["raw_output"]
                
                passed_needle = secret_needle in out_text
                retrieval = 1.0 if passed_needle else (0.5 if "8892" in out_text else 0.0)
                quality = 1.0 if status == "SUCCESS" else 0.0
                
                if status in ("TIMEOUT", "OOM", "FAILED") :
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
                    retrieval_rate, quality_pass_rate, status, publication_class, provenance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, 0, dev, sm, ctx, ctk, 1 if no_kv else 0,
                    prompt_ts if publication_class == "AUTHORITATIVE" else None,
                    eval_ts if publication_class == "AUTHORITATIVE" else None,
                    retrieval if publication_class == "AUTHORITATIVE" else None,
                    quality if publication_class == "AUTHORITATIVE" else None,
                    status,
                    publication_class,
                    "matrix_run:measured" if publication_class == "AUTHORITATIVE" else "matrix_run:unpublished",
                ))
                conn.commit()

    conn.close()
    
    # Export sanitized rows with explicit publication classification.
    conn = sqlite3.connect(db_path)
    exported = export_dashboard(conn)
    conn.close()
    print(f"Exported {exported} rows to docs/benchmarks_data.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()
    
    model_path = args.model_path
    if not model_path or "Program Files" in model_path:
        model_path = f"/home/opencode/llama.cpp/models/{args.model_name}"
        
    run_matrix(args.model_name, model_path)