from pathlib import Path
from unittest.mock import patch
from scripts.run_issue41_ministral_reauthorized import EXPECTED_SHA256, EXPECTED_SIZE_BYTES, MODEL_NAME, configuration, execute, plan

def model(): return {"id":Path(MODEL_NAME).stem,"name":MODEL_NAME,"path":f"/models/{MODEL_NAME}","size_bytes":EXPECTED_SIZE_BYTES}
def test_plan_is_one_job(tmp_path):
 p=plan(model(),tmp_path,600); assert p["expected_job_count"]==1; assert p["declared_context"]==1024; assert p["configuration"]["split_mode"]=="layer"; assert p["configuration"]["tensor_split"]=="1,1"
def test_configuration_rejects_wrong_model_and_size():
 bad=model(); bad["name"]="other.gguf"
 try: configuration(bad)
 except ValueError: pass
 else: raise AssertionError("wrong model accepted")
 bad=model(); bad["size_bytes"]+=1
 try: configuration(bad)
 except ValueError: pass
 else: raise AssertionError("wrong size accepted")
def test_plan_rejects_non_reviewed_timeout(tmp_path):
 try: plan(model(),tmp_path,1200)
 except ValueError as exc: assert "600" in str(exc)
 else: raise AssertionError("non-reviewed timeout accepted")
def test_execute_uses_only_context_1024(tmp_path):
 result={"models":[{"configurations":[{"result":{"status":"SUCCESS"}}]}]}
 with patch("scripts.run_issue41_ministral_reauthorized.execute_suite",return_value=result) as run:
  summary=execute(model(),tmp_path,600)
 assert run.call_args.kwargs["context_sizes"]==[1024]; assert run.call_args.kwargs["performance_context"]==1024; assert summary["job_status"]=="SUCCESS"
def test_binding(): assert EXPECTED_SHA256=="fd46fc371ff0509bfa8657ac956b7de8534d7d9baaa4947975c0648c3aa397f4"
