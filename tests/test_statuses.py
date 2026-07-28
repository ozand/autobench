from src.statuses import boundary_cause, boundary_summary


def test_boundary_cause_maps_known_statuses() -> None:
    assert boundary_cause({"status": "OOM"}) == "BOUNDARY_OOM"
    assert boundary_cause({"status": "REMOTE_TIMEOUT"}) == "BOUNDARY_TIMEOUT"
    assert boundary_cause({"status": "SSH_TIMEOUT"}) == "BOUNDARY_SSH_TIMEOUT"
    assert boundary_cause({"status": "TOKENIZER_ERROR"}) == "BOUNDARY_TOKENIZER_ERROR"
    assert boundary_cause({"status": "UNSUPPORTED_BACKEND"}) == "BOUNDARY_UNSUPPORTED_BACKEND"


def test_boundary_summary_preserves_source_diagnostics() -> None:
    summary = boundary_summary(
        {
            "inconclusive_status": "SSH_TIMEOUT",
            "probes": [
                {"status": "SUCCESS", "context_size": 256},
                {
                    "status": "REMOTE_TIMEOUT",
                    "context_size": 512,
                    "error": "remote command timed out",
                    "return_code": 124,
                },
            ],
        }
    )

    assert summary == {
        "cause_status": "BOUNDARY_TIMEOUT",
        "source_status": "REMOTE_TIMEOUT",
        "context_size": 512,
        "error": "remote command timed out",
        "return_code": 124,
        "inconclusive_status": "SSH_TIMEOUT",
    }
