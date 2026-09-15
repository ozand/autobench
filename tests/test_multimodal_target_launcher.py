import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multimodal_command_plan import APPROVED_ARTIFACTS, APPROVED_IMAGE_DESCRIPTOR, build_first_baseline_command_plan
from src.multimodal_execution_driver import ProcessObservation
from src.multimodal_runner import PreparedMultimodalInvocation
from src.multimodal_target_launcher import (
    DIAGNOSTIC_ARTIFACT_TYPE,
    MultimodalTargetLauncherError,
    _diagnostic_payload,
    _write_new_json,
    launch_one_target_smoke,
    one_shot_process_runner,
    write_sanitized_receipt,
)


def _plan(image_descriptor: dict | None = None) -> dict:
    return build_first_baseline_command_plan(PreparedMultimodalInvocation(
        image_reference=Path("C:/placeholder.png"),
        model_artifact=dict(APPROVED_ARTIFACTS["model_artifact"]),
        projector_artifact=dict(APPROVED_ARTIFACTS["projector_artifact"]),
        image_descriptor=dict(image_descriptor or APPROVED_IMAGE_DESCRIPTOR),
        configuration={"device": "Vulkan0", "split_mode": "none", "split_ratio": None, "context_length": 1024, "cache_type_k": "f16", "cache_type_v": "f16", "max_tokens": 32},
    ))


class FakePipe:
    def __init__(self, data: bytes): self.data, self.closed = data, False
    def read(self, count=-1):
        if not self.data: return b""
        result, self.data = self.data[:count], self.data[count:]
        return result
    def close(self): self.closed = True


class FakeProcess:
    def __init__(self, out=b"ok", err=b"", code=0, timeout=False):
        self.stdout, self.stderr = FakePipe(out), FakePipe(err)
        self.code, self.timeout, self.killed = code, timeout, False
    def wait(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired("binary", timeout)
        return self.code
    def kill(self): self.killed = True


def test_runner_shell_free_bounded_and_consumable_once():
    calls = []
    def factory(argv, **kwargs):
        calls.append((argv, kwargs))
        return FakeProcess(out=b"bounded")
    run = one_shot_process_runner(factory)
    observation = run(["binary", "--flag"], 120)
    assert observation.stdout == b"bounded"
    assert calls[0][0] == ["binary", "--flag"] and calls[0][1]["shell"] is False
    with pytest.raises(RuntimeError): run(["second"], 120)


def test_runner_timeout_and_each_stream_oversize_kill_and_preserve_returncode():
    timeout_process = FakeProcess(timeout=True)
    with patch("src.multimodal_target_launcher.time.monotonic", side_effect=(0, 121)):
        with pytest.raises(TimeoutError):
            one_shot_process_runner(lambda *args, **kwargs: timeout_process)(["binary"], 120)
    assert timeout_process.killed is True
    for out, err in ((b"x" * 9000, b""), (b"", b"x" * 9000)):
        process = FakeProcess(out=out, err=err, code=7)
        observation = one_shot_process_runner(lambda *args, **kwargs: process)(["binary"], 120)
        assert observation.returncode == 7 and len(observation.stdout) == 8193
        assert process.killed is True


def test_runner_rejects_process_creation_failure_and_remains_consumed():
    run = one_shot_process_runner(lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not executable")))
    with pytest.raises(RuntimeError, match="process creation failed"):
        run(["binary"], 120)
    with pytest.raises(RuntimeError, match="already consumed"):
        run(["binary"], 120)


def _strict_receipt() -> dict:
    return {
        "schema_version": 1, "receipt_type": "MULTIMODAL_OCR_SMOKE_EXECUTION",
        "model_artifact": dict(APPROVED_ARTIFACTS["model_artifact"]),
        "projector_artifact": dict(APPROVED_ARTIFACTS["projector_artifact"]),
        "image_descriptor": {"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"},
        "configuration": _plan()["configuration"], "command_family": "multimodal_ocr_runner",
        "terminal_class": "SUCCESS", "task_id": "ocr_smoke_v1", "task_version": 1,
        "output_classification": "OCR_SMOKE_OUTPUT_OBSERVED", "invocation_attempted": True, "inference_invoked": True,
    }


def test_diagnostic_payload_is_bounded_observation_or_unavailable():
    assert _diagnostic_payload(None) == {"artifact_type": DIAGNOSTIC_ARTIFACT_TYPE, "observation": "UNAVAILABLE"}
    payload = _diagnostic_payload(ProcessObservation(7, b"stdout", b"stderr"))
    assert payload == {"artifact_type": DIAGNOSTIC_ARTIFACT_TYPE, "returncode": 7, "stdout": "stdout", "stderr": "stderr", "stdout_truncated": False, "stderr_truncated": False}
    capped = _diagnostic_payload(ProcessObservation(7, b"x" * 9000, b"y" * 9000))
    assert len(capped["stdout"]) == len(capped["stderr"]) == 8192
    assert capped["stdout_truncated"] is capped["stderr_truncated"] is True


def test_diagnostic_writer_is_new_only_and_cleans_failed_temporary(tmp_path, monkeypatch):
    output = tmp_path / "diagnostic.json"
    payload = {"artifact_type": DIAGNOSTIC_ARTIFACT_TYPE, "observation": "UNAVAILABLE"}
    _write_new_json(output, payload, (), ".autobench-diagnostic-")
    assert json.loads(output.read_text()) == payload
    with pytest.raises(MultimodalTargetLauncherError):
        _write_new_json(output, payload, (), ".autobench-diagnostic-")
    failed = tmp_path / "failed.json"
    monkeypatch.setattr("src.multimodal_target_launcher.os.link", lambda *args: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(MultimodalTargetLauncherError, match="write failed"):
        _write_new_json(failed, payload, (), ".autobench-diagnostic-")
    assert not failed.exists() and not list(tmp_path.glob(".autobench-diagnostic-*"))


def test_diagnostic_writer_sanitizes_temporary_creation_failure(tmp_path, monkeypatch):
    payload = {"artifact_type": DIAGNOSTIC_ARTIFACT_TYPE, "observation": "UNAVAILABLE"}
    monkeypatch.setattr("src.multimodal_target_launcher.tempfile.mkstemp", lambda **kwargs: (_ for _ in ()).throw(OSError("creation failed")))
    with pytest.raises(MultimodalTargetLauncherError, match="diagnostic output write failed"):
        _write_new_json(tmp_path / "diagnostic.json", payload, (), ".autobench-diagnostic-")


def test_diagnostic_writer_sanitizes_cleanup_failure(tmp_path, monkeypatch):
    output = tmp_path / "diagnostic.json"
    payload = {"artifact_type": DIAGNOSTIC_ARTIFACT_TYPE, "observation": "UNAVAILABLE"}
    original_unlink = Path.unlink
    monkeypatch.setattr("src.multimodal_target_launcher.os.link", lambda *args: (_ for _ in ()).throw(OSError("publication failed")))
    monkeypatch.setattr(Path, "unlink", lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup failed")) if self.name.startswith(".autobench-diagnostic-") else original_unlink(self, *args, **kwargs))
    with pytest.raises(MultimodalTargetLauncherError, match="diagnostic output write failed"):
        _write_new_json(output, payload, (), ".autobench-diagnostic-")
    assert not output.exists()


def test_safe_writer_validates_receipt_and_refuses_collision_or_existing_output(tmp_path):
    receipt = _strict_receipt()
    output = tmp_path / "receipt.json"
    write_sanitized_receipt(output, receipt, ())
    assert json.loads(output.read_text())["terminal_class"] == "SUCCESS"
    with pytest.raises(MultimodalTargetLauncherError): write_sanitized_receipt(output, receipt, ())
    protected = tmp_path / "model.gguf"; protected.write_bytes(b"model")
    with pytest.raises(MultimodalTargetLauncherError): write_sanitized_receipt(protected, receipt, (protected,))


def test_launch_rejects_output_under_transient_root_before_wrapper(tmp_path):
    with patch("src.multimodal_target_launcher.run_one_target_smoke") as wrapper:
        with pytest.raises(MultimodalTargetLauncherError, match="outside the temporary root"):
            launch_one_target_smoke(
                binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "model.gguf", projector_path=tmp_path / "projector.gguf",
                target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=tmp_path, output=tmp_path / "receipt.json",
            )
    wrapper.assert_not_called()


def test_safe_writer_cleans_private_temporary_on_publication_failure(tmp_path, monkeypatch):
    receipt = _strict_receipt()
    monkeypatch.setattr("src.multimodal_target_launcher.os.link", lambda *args: (_ for _ in ()).throw(OSError("publication failed")))
    with pytest.raises(OSError):
        write_sanitized_receipt(tmp_path / "receipt.json", receipt, ())
    assert not (tmp_path / "receipt.json").exists()
    assert not list(tmp_path.glob(".autobench-receipt-*"))


def test_launch_integrates_wrapper_with_actual_strict_receipt_persistence(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    output = tmp_path / "receipt.json"
    with patch("src.multimodal_target_launcher.run_one_target_smoke", return_value=_strict_receipt()) as wrapper:
        result = launch_one_target_smoke(
            binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
            target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=temporary_root, output=output,
        )
    wrapper.assert_called_once()
    assert result["terminal_class"] == json.loads(output.read_text())["terminal_class"] == "SUCCESS"


def test_real_wrapper_and_runner_write_nonzero_diagnostic(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    image = _cli_jpeg(tmp_path)
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    process = FakeProcess(out=b"runtime-out", err=b"runtime-err", code=7)
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=(dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"]))):
        receipt = launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=image, plan=_plan({"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"}), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic, popen_factory=lambda *args, **kwargs: process)
    assert receipt["terminal_class"] == "EXECUTION_ERROR"
    assert receipt["output_classification"] == "RUNTIME_NONZERO"
    payload = json.loads(diagnostic.read_text())
    assert payload == {"artifact_type": DIAGNOSTIC_ARTIFACT_TYPE, "returncode": 7, "stdout": "runtime-out", "stderr": "runtime-err", "stdout_truncated": False, "stderr_truncated": False}
    assert process.stdout.closed and process.stderr.closed


def test_real_wrapper_and_runner_write_overflow_diagnostic(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    image = _cli_jpeg(tmp_path)
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    process = FakeProcess(out=b"x" * 9000, code=7)
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=(dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"]))):
        receipt = launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=image, plan=_plan({"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"}), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic, popen_factory=lambda *args, **kwargs: process)
    payload = json.loads(diagnostic.read_text())
    assert receipt["output_classification"] == "OUTPUT_OVERSIZE"
    assert payload["returncode"] == 7 and payload["stdout_truncated"] is True and len(payload["stdout"]) == 8192
    assert process.killed is True


def test_real_wrapper_and_runner_timeout_writes_unavailable_diagnostic(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    image = _cli_jpeg(tmp_path)
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    process = FakeProcess(timeout=True)
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=(dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"]))), patch("src.multimodal_target_launcher.time.monotonic", side_effect=(0, 121)):
        receipt = launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=image, plan=_plan({"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"}), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic, popen_factory=lambda *args, **kwargs: process)
    assert receipt["output_classification"] == "RUNTIME_TIMEOUT"
    assert json.loads(diagnostic.read_text())["observation"] == "UNAVAILABLE"
    assert process.killed is True


def test_diagnostic_writer_failure_propagates_after_strict_receipt(tmp_path, monkeypatch):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    image = _cli_jpeg(tmp_path)
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    process = FakeProcess(code=7)
    monkeypatch.setattr("src.multimodal_target_launcher._write_new_json", lambda *args: (_ for _ in ()).throw(MultimodalTargetLauncherError("diagnostic output write failed")))
    with patch("src.multimodal_target_wrapper.validate_target_identity", return_value=(dict(APPROVED_ARTIFACTS["model_artifact"]), dict(APPROVED_ARTIFACTS["projector_artifact"]))):
        with pytest.raises(MultimodalTargetLauncherError, match="diagnostic output write failed"):
            launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=image, plan=_plan({"format": "jpg", "width": 28, "height": 28, "byte_class": "small", "validation_status": "VALID"}), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic, popen_factory=lambda *args, **kwargs: process)
    assert output.exists() and not diagnostic.exists()


def test_launch_rejects_diagnostic_output_under_transient_root_before_wrapper(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    with patch("src.multimodal_target_launcher.run_one_target_smoke") as wrapper:
        with pytest.raises(MultimodalTargetLauncherError, match="diagnostic output must be outside the temporary root"):
            launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "model.gguf", projector_path=tmp_path / "projector.gguf", target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=temporary_root, output=tmp_path / "receipt.json", diagnostic_output=temporary_root / "diagnostic.json")
    wrapper.assert_not_called()


def test_launch_timeout_writes_unavailable_diagnostic(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    def wrapper(**kwargs):
        kwargs["diagnostic_observer"](None)
        receipt = _strict_receipt(); receipt.update({"terminal_class": "INCONCLUSIVE", "output_classification": "RUNTIME_TIMEOUT", "inference_invoked": False})
        return receipt
    with patch("src.multimodal_target_launcher.run_one_target_smoke", side_effect=wrapper):
        launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic)
    assert json.loads(diagnostic.read_text())["observation"] == "UNAVAILABLE"


def test_launch_writes_bounded_overflow_diagnostic(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    def wrapper(**kwargs):
        kwargs["diagnostic_observer"](ProcessObservation(7, b"x" * 9000, b""))
        receipt = _strict_receipt(); receipt.update({"terminal_class": "METRIC_PARSE_FAILED", "output_classification": "OUTPUT_OVERSIZE", "inference_invoked": False})
        return receipt
    with patch("src.multimodal_target_launcher.run_one_target_smoke", side_effect=wrapper):
        launch_one_target_smoke(binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic)
    payload = json.loads(diagnostic.read_text())
    assert payload["stdout_truncated"] is True and len(payload["stdout"]) == 8192


def test_launch_writes_one_diagnostic_bound_to_wrapper_observation(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    output, diagnostic = tmp_path / "receipt.json", tmp_path / "diagnostic.json"
    def wrapper(**kwargs):
        kwargs["diagnostic_observer"](ProcessObservation(7, b"bounded-out", b"bounded-err"))
        return _strict_receipt()
    with patch("src.multimodal_target_launcher.run_one_target_smoke", side_effect=wrapper):
        launch_one_target_smoke(
            binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf",
            target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=temporary_root, output=output, diagnostic_output=diagnostic,
        )
    assert json.loads(diagnostic.read_text())["returncode"] == 7
    assert json.loads(diagnostic.read_text())["stderr"] == "bounded-err"


def test_launch_integrates_one_mock_wrapper_call_and_one_safe_receipt(tmp_path):
    temporary_root = tmp_path / "temporary"; temporary_root.mkdir()
    output = tmp_path / "receipt.json"
    receipt = {"sentinel": "sanitized"}
    with patch("src.multimodal_target_launcher.run_one_target_smoke", return_value=receipt) as wrapper, patch("src.multimodal_target_launcher.write_sanitized_receipt") as writer, patch("subprocess.Popen") as real:
        result = launch_one_target_smoke(
            binary_path=tmp_path / "llama-mtmd-cli", model_path=tmp_path / "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", projector_path=tmp_path / "mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf", target_image_path=tmp_path / "document.jpg", plan=_plan(), temporary_root=temporary_root, output=output, popen_factory=lambda *args, **kwargs: FakeProcess(),
        )
    assert result is receipt
    wrapper.assert_called_once(); writer.assert_called_once(); real.assert_not_called()


def _load_cli_module():
    path = Path("scripts/run_ocr_target_smoke.py")
    spec = importlib.util.spec_from_file_location("issue147_ocr_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _cli_jpeg(tmp_path):
    image = tmp_path / "document.jpg"
    image.write_bytes(b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x1c\x00\x1c\x03" + b"\x00" * 10 + b"\xff\xd9")
    return image


def test_cli_delegates_arguments_without_launching_process(monkeypatch, tmp_path, capsys):
    cli = _load_cli_module()
    image = _cli_jpeg(tmp_path)
    captured = {}
    def fake_launch(**kwargs):
        captured.update(kwargs)
        return {"terminal_class": "INCONCLUSIVE", "inference_invoked": False}
    monkeypatch.setattr(cli, "launch_one_target_smoke", fake_launch)
    monkeypatch.setattr(sys, "argv", ["run_ocr_target_smoke.py", "--binary", "binary", "--model", "model", "--projector", "projector", "--image", str(image), "--temporary-root", str(tmp_path), "--output", str(tmp_path / "receipt.json")])
    assert cli.main() == 0
    assert captured["binary_path"] == Path("binary")
    assert json.loads(capsys.readouterr().out)["terminal_class"] == "INCONCLUSIVE"


def test_cli_maps_wrapper_error_to_sanitized_parser_error(monkeypatch, tmp_path):
    cli = _load_cli_module()
    image = _cli_jpeg(tmp_path)
    from src.multimodal_target_wrapper import MultimodalTargetWrapperError
    monkeypatch.setattr(cli, "launch_one_target_smoke", lambda **kwargs: (_ for _ in ()).throw(MultimodalTargetWrapperError("bad input")))
    monkeypatch.setattr(sys, "argv", ["run_ocr_target_smoke.py", "--binary", "binary", "--model", "model", "--projector", "projector", "--image", str(image), "--temporary-root", str(tmp_path), "--output", str(tmp_path / "receipt.json")])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2


def test_launcher_source_has_no_shell_or_transport_primitive():
    source = Path("src/multimodal_target_launcher.py").read_text(encoding="utf-8")
    for forbidden in ("run_host_command", "requests", "os.system", "bash -lc"):
        assert forbidden not in source
