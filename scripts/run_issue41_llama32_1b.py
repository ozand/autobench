#!/usr/bin/env python3
"""Run the reviewed bounded Llama 3.2 1B follow-up for Issue #41."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from authoritative_bench import FULL_POLICY, SINGLE_GPU_FIT_BYTES, discover_remote_models, execute_suite
from src.protocol_receipt import load_and_validate_receipt
from src.remote import run_host_command
from src.statuses import sanitize_artifact
MODEL_NAME="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
EXPECTED_SIZE_BYTES=807694368
EXPECTED_SHA256="3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1"
PRIMARY_TIMEOUT=600
EXPECTED_JOB_COUNT=3
DEFAULT_RECEIPT=Path("results/receipts/Llama-3.2-1B-Instruct-Q4_K_M.issue41.json")
DEFAULT_OUTPUT_DIR=Path("results/runs/issue41-llama32-1b")

def configurations(model):
    if model["name"] != MODEL_NAME: raise ValueError("unexpected model basename")
    if model["size_bytes"] != EXPECTED_SIZE_BYTES: raise ValueError("target GGUF size does not match reviewed artifact")
    base={"mode":"full","cache_type_k":"f16","cache_type_v":"f16","no_kv_offload":False}
    return [{**base,"device":"Vulkan0","tensor_split":None,"split_mode":"none"},{**base,"device":"Vulkan1","tensor_split":None,"split_mode":"none"},{**base,"device":"Vulkan0,Vulkan1","tensor_split":"1,1","split_mode":"layer"}]

def verify_artifact():
    r=run_host_command(f"sha256sum /home/opencode/llama.cpp/models/{MODEL_NAME} | cut -d ' ' -f1",timeout=120)
    if r.returncode != 0 or r.stdout.strip() != EXPECTED_SHA256: raise ValueError("target GGUF checksum does not match reviewed artifact")

def reviewed_plan(model, output_dir, timeout):
    if timeout != PRIMARY_TIMEOUT: raise ValueError("timeout must remain the reviewed primary value of 600 seconds")
    return {"schema_version":1,"issue":41,"provenance":"Source: governed remaining-model follow-up under #41","mode":"issue41_llama32_1b_followup","execution_status":"planned_not_run","generated_at":datetime.now(timezone.utc).isoformat(),"model":{"basename":model["name"],"size_bytes":model["size_bytes"],"quantization":"Q4_K_M","backend":"vulkan"},"configurations":configurations(model),"policy":{"contexts_single_gpu":[1024],"contexts_dual_layer":[1024,2048,4096],"cache_type_k":"f16","cache_type_v":"f16","no_kv_offload":False,"prompt_tokens":512,"output_tokens":64,"boundary_output_tokens":16,"retrieval_positions":FULL_POLICY["retrieval_positions"],"retrieval_repetitions":5,"quality_tasks":2,"performance_warmups":1,"performance_repetitions":3,"reliability_threshold":0.8,"timeout_seconds":timeout},"expected_job_count":3,"output_dir":str(output_dir),"stop_rule":"Stop at the first unexpected result and classify it before further inference.","inference_authorized":False}

def suite_plan(model, config):
    return {"schema_version":1,"mode":"suite","authoritative":False,"execution_status":"planned_not_run","generated_at":datetime.now(timezone.utc).isoformat(),"host":"k7000","gpu_memory_bytes_per_device":2*1024**3,"single_gpu_fit_threshold_bytes":SINGLE_GPU_FIT_BYTES,"policy":dict(FULL_POLICY),"models":[{"id":Path(model["name"]).stem,"name":model["name"],"path":model["path"],"size_bytes":model["size_bytes"],"configurations":[config]}]}

def execute_job(model, config, output_dir, index, total):
    contexts=[1024,2048,4096] if config["device"]=="Vulkan0,Vulkan1" else [1024]
    result=execute_suite(suite_plan(model,config),timeout=PRIMARY_TIMEOUT,context_sizes=contexts,boundary_step=256,retrieval_repetitions=5,reliability_threshold=0.8,performance_context=1024,prompt_tokens=512,output_tokens=64,warmups=1,performance_repetitions=3,dataset_dir="datasets/validation",max_tasks=2)
    data=result["models"][0]["configurations"][0].get("result",{})
    job={"job_index":index,"total_jobs":total,"device":config["device"],"split_mode":config["split_mode"],"tensor_split":config["tensor_split"],"status":data.get("status","EXECUTION_ERROR"),"result":result}
    (output_dir/f"job_{index:02d}.json").write_text(json.dumps(sanitize_artifact(job),indent=2)+"\n",encoding="utf-8")
    return job

def execute(model, output_dir):
    output_dir.mkdir(parents=True,exist_ok=True); configs=configurations(model); jobs=[]; terminal="SUCCESS"
    for i,config in enumerate(configs,1):
        job=execute_job(model,config,output_dir,i,len(configs)); jobs.append(job)
        if job["status"] != "SUCCESS": terminal=job["status"]; break
    summary={"schema_version":1,"issue":41,"provenance":"Source: governed remaining-model follow-up under #41","mode":"issue41_llama32_1b_followup","authoritative":False,"generated_at":datetime.now(timezone.utc).isoformat(),"model":{"basename":MODEL_NAME,"size_bytes":EXPECTED_SIZE_BYTES,"sha256":EXPECTED_SHA256,"quantization":"Q4_K_M","backend":"vulkan"},"configurations":configs,"execution_status":terminal,"jobs_completed":len(jobs),"jobs_expected":len(configs),"jobs":jobs}
    (output_dir/"summary.json").write_text(json.dumps(sanitize_artifact(summary),indent=2)+"\n",encoding="utf-8"); return summary

def main():
    p=argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True); mode.add_argument("--dry-run",action="store_true"); mode.add_argument("--execute",action="store_true"); p.add_argument("--models",required=True); p.add_argument("--receipt",type=Path,required=True); p.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR); p.add_argument("--timeout",type=int,default=PRIMARY_TIMEOUT); a=p.parse_args()
    if a.models!=MODEL_NAME: p.error("--models must name exact Llama GGUF")
    if a.receipt!=DEFAULT_RECEIPT: p.error("--receipt must name Issue #41 receipt")
    if a.timeout!=PRIMARY_TIMEOUT: p.error("--timeout must remain 600")
    matches=[m for m in discover_remote_models() if m["name"]==MODEL_NAME]
    if len(matches)!=1: p.error("exact target GGUF was not found exactly once")
    model=matches[0]
    try:
        validation=load_and_validate_receipt(a.receipt,model_name=MODEL_NAME,backend="vulkan",governing_issue=41,check_files_exist=True,expected_job_count=EXPECTED_JOB_COUNT,expected_artifact={"basename":MODEL_NAME,"size_bytes":EXPECTED_SIZE_BYTES,"sha256":EXPECTED_SHA256},expected_configurations=configurations(model))
        if validation["status"]!="PROTOCOL_RECEIPT_VALID": p.error("protocol receipt validation failed")
        verify_artifact()
    except (ValueError,OSError) as e: p.error(str(e))
    if a.dry_run:
        a.output_dir.mkdir(parents=True,exist_ok=True); out=a.output_dir/"dry_run_plan.json"; out.write_text(json.dumps(reviewed_plan(model,a.output_dir,a.timeout),indent=2)+"\n",encoding="utf-8"); print("Issue #41 Llama 3.2 1B dry-run: jobs=3; contexts=[1024, 2048, 4096]; inference=0"); print(f"Plan: {out}"); return 0
    s=execute(model,a.output_dir); print(json.dumps({"issue":41,"model":MODEL_NAME,"job_status":s["execution_status"],"jobs_completed":s["jobs_completed"],"jobs_expected":s["jobs_expected"]},sort_keys=True)); return 0 if s["execution_status"]=="SUCCESS" else 3
if __name__=="__main__": sys.exit(main())
