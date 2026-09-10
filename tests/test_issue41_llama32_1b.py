from pathlib import Path
from unittest.mock import patch
from scripts.run_issue41_llama32_1b import EXPECTED_SHA256, EXPECTED_SIZE_BYTES, MODEL_NAME, configurations, execute, reviewed_plan

def model():
    return {"id":Path(MODEL_NAME).stem,"name":MODEL_NAME,"path":f"/models/{MODEL_NAME}","size_bytes":EXPECTED_SIZE_BYTES}

def test_plan_has_three_reviewed_jobs(tmp_path):
    p=reviewed_plan(model(),tmp_path,600); assert p["expected_job_count"]==3; assert len(p["configurations"])==3; assert p["policy"]["contexts_dual_layer"]==[1024,2048,4096]

def test_configurations_include_single_and_dual_layer():
    c=configurations(model()); assert [x["device"] for x in c]==["Vulkan0","Vulkan1","Vulkan0,Vulkan1"]; assert c[2]["split_mode"]=="layer"; assert c[2]["tensor_split"]=="1,1"

def test_wrong_model_or_size_fails_closed():
    bad=model(); bad["name"]="other.gguf"
    try: configurations(bad)
    except ValueError: pass
    else: raise AssertionError("wrong model accepted")
    bad=model(); bad["size_bytes"]+=1
    try: configurations(bad)
    except ValueError: pass
    else: raise AssertionError("wrong size accepted")

def test_execute_uses_context_contract(tmp_path):
    result={"models":[{"configurations":[{"result":{"status":"SUCCESS"}}]}]}
    with patch("scripts.run_issue41_llama32_1b.execute_suite",return_value=result) as run:
        summary=execute(model(),tmp_path)
    assert run.call_args_list[0].kwargs["context_sizes"]==[1024]
    assert run.call_args_list[1].kwargs["context_sizes"]==[1024]
    assert run.call_args_list[2].kwargs["context_sizes"]==[1024,2048,4096]
    assert summary["jobs_completed"]==3

def test_binding(): assert EXPECTED_SHA256=="3f5a22426976ab26cfe84dba63c1d08391717abb1af893e10f1b2968d862dcc1"
