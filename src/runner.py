import subprocess
import re
import json
import time
import urllib.request
import os
import shlex


class Runner:
    @staticmethod
    def count_local_tokens(model_path: str, text: str, timeout: int = 60) -> int:
        """Count text tokens with the tokenizer embedded in a remote GGUF model."""
        command = (
            "timeout "
            f"{timeout}s /home/opencode/llama.cpp/build/bin/llama-tokenize "
            f"-m {shlex.quote(model_path)} --stdin --show-count"
        )
        try:
            result = subprocess.run(
                ["ssh", "opencode@192.168.1.171", command],
                input=text,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Tokenizer SSH execution timed out") from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"Tokenizer failed with code {result.returncode}: {result.stderr.strip()}"
            )

        match = re.search(r"Total number of tokens:\s*(\d+)", result.stdout)
        if not match:
            raise RuntimeError("Tokenizer did not return a token count")
        return int(match.group(1))

    @staticmethod
    def get_litellm_config():
        config_path = "/home/ozand/.pi/agent/models.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, "r") as f:
            data = json.load(f)
        litellm_data = data.get("providers", {}).get("litellm", {})
        base_url = litellm_data.get("baseUrl")
        api_key = litellm_data.get("apiKey")
        return base_url, api_key

    @staticmethod
    def run_local_vulkan(
        prompt: str,
        max_tokens: int = 128,
        device: str = "Vulkan0",
        model_path: str = "/home/opencode/llama.cpp/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        timeout: int = 60,
        context_length: int = 1024,
        ts_split: str = None,
    ) -> dict:
        """
        Executes llama-cli on k7000 via SSH and parses results.
        """
        # Escaping quotes for SSH execution
        escaped_prompt = prompt.replace("'", "'\"'\"'")
        if ts_split:
            ts_flag = f"-ts {ts_split} "
        elif "," in device:
            ts_flag = "-ts 1,1 "
        else:
            ts_flag = ""
        cmd = (
            f"timeout {timeout}s /home/opencode/llama.cpp/build/bin/llama-cli "
            f"-m '{model_path}' "
            f"-ngl 99 -dev {device} {ts_flag}-c {context_length} -p '{escaped_prompt}' "
            f"-n {max_tokens} -st -no-cnv --no-display-prompt --simple-io < /dev/null"
        )

        ssh_cmd = ["ssh", "opencode@192.168.1.171", cmd]
        command_args = shlex.split(cmd)

        start_time = time.time()
        try:
            res = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "status": "SSH_TIMEOUT",
                "error": f"SSH execution timed out after {timeout + 10} seconds",
                "return_code": None,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "raw_output": exc.stdout or "",
                "elapsed_seconds": elapsed,
                "command_args": command_args,
            }

        elapsed = time.time() - start_time
        if res.returncode != 0:
            status = Runner._classify_local_failure(res.returncode, res.stdout, res.stderr)
            return {
                "success": False,
                "status": status,
                "error": Runner._failure_message(status, res.returncode, res.stderr),
                "return_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "raw_output": res.stdout,
                "elapsed_seconds": elapsed,
                "command_args": command_args,
            }

        stdout = res.stdout

        # Regex to parse generation speed
        # e.g. [ Prompt: 287.8 t/s | Generation: 36.7 t/s ]
        speed_match = re.search(
            r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
            stdout,
        )
        prompt_ts = 0.0
        gen_ts = 0.0
        if speed_match:
            prompt_ts = float(speed_match.group(1))
            gen_ts = float(speed_match.group(2))

        # Extract response text
        # llama-cli prints the generated response immediately before '[ Prompt: XX t/s | Generation: YY t/s ]'
        # or before 'Exiting...'
        lines = stdout.splitlines()
        end_idx = -1
        for i, line in enumerate(lines):
            if "[ Prompt:" in line or "Exiting..." in line:
                end_idx = i
                break

        if end_idx != -1:
            # Find the last line before end_idx that contains the prompt termination marker
            start_idx = 0
            for i in range(end_idx - 1, -1, -1):
                if "--- END OF PROMPT ---" in lines[i]:
                    start_idx = i + 1
                    break
            else:
                # Long prompts may be abbreviated by llama-cli as "... (truncated)".
                for i in range(end_idx - 1, -1, -1):
                    if lines[i].rstrip().endswith("... (truncated)"):
                        start_idx = i + 1
                        break
                else:
                    prompt_clean = prompt.strip()
                    prompt_suffix = prompt_clean[-30:] if len(prompt_clean) > 30 else prompt_clean
                    for i in range(end_idx - 1, -1, -1):
                        if prompt_suffix in lines[i]:
                            start_idx = i + 1
                            break
                    else:
                        for i in range(end_idx - 1, -1, -1):
                            if lines[i].startswith("> "):
                                start_idx = i + 1
                                break

            content_lines = lines[start_idx:end_idx]
            response_text = "\n".join(content_lines).strip()
        else:
            response_text = stdout.strip()

        return {
            "success": True,
            "status": "SUCCESS",
            "response": response_text,
            "prompt_speed_ts": prompt_ts,
            "generation_speed_ts": gen_ts,
            "elapsed_seconds": elapsed,
            "tokens_approx": len(response_text.split()) * 1.33,
            "return_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "command_args": command_args,
        }

    @staticmethod
    def _classify_local_failure(return_code: int, stdout: str, stderr: str) -> str:
        """Map llama.cpp and transport failures to stable result statuses."""
        if return_code == 124:
            return "REMOTE_TIMEOUT"

        diagnostic = f"{stdout}\n{stderr}".lower()
        if "does not support split buffers" in diagnostic:
            return "UNSUPPORTED_BACKEND"
        if "erroroutofdevicememory" in diagnostic or "out of device memory" in diagnostic:
            return "OOM"
        if "exceeds the available context size" in diagnostic:
            return "CONTEXT_OVERFLOW"
        if "failed to load model" in diagnostic or "model loading error" in diagnostic:
            return "MODEL_LOAD_ERROR"
        if return_code == 255:
            return "SSH_ERROR"
        return "EXECUTION_ERROR"

    @staticmethod
    def _failure_message(status: str, return_code: int, stderr: str) -> str:
        """Build a concise diagnostic without discarding the original stderr field."""
        detail = stderr.strip().splitlines()[-1] if stderr.strip() else "no stderr"
        return f"{status} (code {return_code}): {detail}"

    @staticmethod
    def run_frontier_api(model: str, prompt: str, max_tokens: int = 128) -> dict:
        """
        Executes frontier model via LiteLLM proxy.
        """
        base_url, api_key = Runner.get_litellm_config()
        if not base_url or not api_key:
            return {
                "success": False,
                "error": "LiteLLM configurations are missing in models.json",
            }

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        req = urllib.request.Request(
            url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST"
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                res_body = json.loads(response.read().decode("utf-8"))
            end_time = time.time()
            elapsed = end_time - start_time

            choices = res_body.get("choices", [])
            if not choices:
                return {"success": False, "error": f"Empty response: {res_body}"}

            choice = choices[0]
            message = choice.get("message", {})
            response_text = message.get("content", "").strip()

            usage = res_body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            prompt_ts = (
                prompt_tokens / (elapsed * 0.2) if elapsed > 0 else 0.0
            )  # Estimate prompt speed
            gen_ts = (
                completion_tokens / (elapsed * 0.8) if elapsed > 0 else 0.0
            )  # Estimate generation speed

            return {
                "success": True,
                "response": response_text,
                "prompt_speed_ts": prompt_ts,
                "generation_speed_ts": gen_ts,
                "elapsed_seconds": elapsed,
                "tokens_approx": completion_tokens,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
