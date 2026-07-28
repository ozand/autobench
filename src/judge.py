import re
import json


class Judge:
    @staticmethod
    def strip_thinking(text: str) -> str:
        """
        Removes chain-of-thought and thinking blocks (XML, brackets, or raw).
        """
        if not text:
            return ""
        # Remove XML style think blocks: <think>...</think> or <think>... (if truncated)
        text = re.sub(r'(?s)<think>.*?(?:</think>|\Z)', '', text)
        # Remove bracket style think blocks: [Start thinking]...[End thinking] or [Start thinking]...
        text = re.sub(r'(?s)\[Start thinking\].*?(?:\[End thinking\]|\Z)', '', text)
        # Remove raw thinking process text: Thinking Process:... until double newline or end
        text = re.sub(r'(?s)Thinking Process:.*?(?:\n\n|\Z)', '', text)
        return text.strip()

    @staticmethod
    def validate_deterministic(response: str, constraints: dict) -> dict:
        """
        Runs regex, json schema, and max length checks.
        """
        cleaned_response = Judge.strip_thinking(response)
        issues = []
        passed = True

        # 1. Max length check
        max_length = constraints.get("max_length")
        if max_length and len(cleaned_response) > max_length:
            passed = False
            issues.append(
                f"Response length ({len(cleaned_response)}) exceeds max allowed ({max_length})"
            )

        # 2. Banned phrases check
        banned_phrases = constraints.get("banned_phrases", [])
        for phrase in banned_phrases:
            if phrase in cleaned_response:
                passed = False
                issues.append(f"Banned phrase detected: '{phrase}'")

        # 3. Required regex check
        required_regex = constraints.get("required_regex", [])
        for regex_str in required_regex:
            if not re.search(regex_str, cleaned_response, re.IGNORECASE):
                passed = False
                issues.append(f"Required pattern '{regex_str}' not matched")

        # 4. JSON Schema check
        json_schema = constraints.get("json_schema")
        if json_schema:
            # Try to extract JSON from response
            try:
                # Find the first occurrences of { and }
                start = cleaned_response.find("{")
                end = cleaned_response.rfind("}")
                if start != -1 and end != -1:
                    json_str = cleaned_response[start : end + 1]
                else:
                    json_str = cleaned_response

                parsed_json = json.loads(json_str)

                # Basic schema validation
                item_passed, err = Judge._validate_json_schema(parsed_json, json_schema)
                if not item_passed:
                    passed = False
                    issues.append(f"JSON Schema mismatch: {err}")
            except json.JSONDecodeError as jde:
                passed = False
                issues.append(f"Failed to parse JSON from response: {str(jde)}")

        return {"passed": passed, "issues": issues, "score": 1.0 if passed else 0.0}

    @staticmethod
    def _validate_json_schema(data: dict, schema: dict) -> tuple[bool, str]:
        """
        Simple validator to avoid importing jsonschema package if not installed.
        """
        # Validate type
        schema_type = schema.get("type", "object")
        if schema_type == "object" and not isinstance(data, dict):
            return False, f"Expected dict, got {type(data).__name__}"

        # Validate required properties
        required = schema.get("required", [])
        for req in required:
            if req not in data:
                return False, f"Missing required parameter '{req}'"

        # Validate properties
        properties = schema.get("properties", {})
        for prop, prop_schema in properties.items():
            if prop in data:
                val = data[prop]
                prop_type = prop_schema.get("type")
                if prop_type == "string" and not isinstance(val, str):
                    return (
                        False,
                        f"Property '{prop}' expects string, got {type(val).__name__}",
                    )
                elif prop_type == "boolean" and not isinstance(val, bool):
                    return (
                        False,
                        f"Property '{prop}' expects boolean, got {type(val).__name__}",
                    )
                elif prop_type == "integer" and not isinstance(val, int):
                    return (
                        False,
                        f"Property '{prop}' expects integer, got {type(val).__name__}",
                    )
                elif prop_type == "number" and not isinstance(val, (int, float)):
                    return (
                        False,
                        f"Property '{prop}' expects number, got {type(val).__name__}",
                    )

                # Check enum
                if "enum" in prop_schema and val not in prop_schema["enum"]:
                    return (
                        False,
                        f"Property '{prop}' value '{val}' not in allowed enum list {prop_schema['enum']}",
                    )

        return True, ""

    @staticmethod
    def run_llm_judge(
        runner_instance, target_model: str, task: str, prompt: str, response: str
    ) -> dict:
        """
        Uses a frontier model via Runner to grade response quality from 1 to 5.
        """
        judge_prompt = f"""
You are an expert evaluator judge. Score the given text generated by an assistant for a target prompt.
Rate the quality from 1 (terrible, incorrect, or did not follow instructions) to 5 (perfect, followed all options).

TASK INFO: {task}
USER PROMPT:
---
{prompt}
---

ASSISTANT RESPONSE:
---
{response}
---

Provide a score between 1 and 5 and a short reason.
Output JSON only in this format:
{{"score": 5, "reason": "Explanation here"}}
"""
        # Call frontier API model to judge (we can class-method link to runner_instance)
        try:
            res_dict = runner_instance.run_frontier_api(
                model="cl/gpt-5.6-terra",  # High-quality judge model
                prompt=judge_prompt,
                max_tokens=200,
            )
            if not res_dict.get("success"):
                # Fallback to gemini if gpt-5.6 is failing/slow
                res_dict = runner_instance.run_frontier_api(
                    model="cl/gemini-2.5-flash", prompt=judge_prompt, max_tokens=200
                )

            if res_dict.get("success"):
                text = res_dict.get("response", "")
                # Find JSON bounds
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    parsed = json.loads(text[start : end + 1])
                    return {
                        "success": True,
                        "judge_score": float(parsed.get("score", 3.0)),
                        "reason": parsed.get("reason", "No reason provided."),
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Failed to extract JSON from judge response: {text}",
                    }
            else:
                return {"success": False, "error": res_dict.get("error")}
        except Exception as e:
            return {"success": False, "error": str(e)}
