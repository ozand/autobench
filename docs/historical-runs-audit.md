# Historical Runs Audit & Protocol Ledger

**Document status: active audit ledger for Issue #1 umbrella benchmark execution.**
**Governing issue:** Issue #23 (`[B] audit historical qwen2.5-0.5b-q8_0 and Llama-3.2-1B runs before resuming Issue #1`).
**Parent issue:** Issue #1 (`test(k7000-autobench): rebuild the authoritative local GGUF benchmark matrix`).
**Protocol specification:** `docs/model-testing-protocol.md`.

---

## 1. Executive Summary & Policy Gate

The AutoBench authoritative GGUF benchmark matrix requires every published model configuration to complete the strict four-stage protocol:
1. **Stage 1: Deep Research (Surf CLI)** — authoritative architecture, quantization, context limits, and llama.cpp/Vulkan flags.
2. **Stage 2: Knowledge Base & Verification Gate (OKF/QMD & Receipt)** — verified wiki entry and fail-closed JSON receipt (`results/receipts/<model>.json`).
3. **Stage 3: Reviewed Staged Benchmark Execution** — single-GPU baseline (`Vulkan0`, `Vulkan1`), dual-GPU layer split (`-sm layer`, e.g. `1,2` / `2,3`), and context/KV cache bounds.
4. **Stage 4: Evidence-First Classification & Publication** — authoritative classification, error taxonomy matching, and synchronized sanitized artifacts.

### One-Model-At-A-Time Execution Gate
- No model may be executed or planned in batch to "clear inventory".
- The next model in inventory order cannot begin Stage 1 or Stage 3 until the preceding model is explicitly completed through Stage 4 or classified with a terminal taxonomy disposition.
- A successful `--dry-run` or historical `completed_suite` status does **not** grant authoritative status without the required Stage 1-4 evidence receipts.

---

## 2. Historical Runs Audit Ledger

| Model ID | Historical Artifacts | Stage 1 (Research) | Stage 2 (Receipt/KB) | Stage 3 (Staged Runs) | Stage 4 (Classification) | Authoritative Status | Next Required Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qwen2.5-0.5b-instruct-q8_0` | `results/inventory/qwen2.5-0.5b-instruct-q8_0_Vulkan0_none_full.json`, `...Vulkan1_none_full.json` | ⚠️ Incomplete (only model card raw) | ⚠️ Missing receipt; KB entry exists (`QWEN-001`) | ⚠️ Partial (single-GPU run, missing dual-GPU `-sm layer` matrix) | ⚠️ Partial failure (retrieval rate 46.7%, SSH boundary timeout) | **NON-AUTHORITATIVE** | Full Stage 1-4 rerun under dedicated model issue |
| `Llama-3.2-1B-Instruct-Q4_K_M` | `results/inventory/Llama-3.2-1B-Instruct-Q4_K_M_Vulkan0_none_full.json`, `...Vulkan1_none_full.json` | ❌ Missing | ❌ Missing receipt & KB entry | ⚠️ Partial (retrieval rate 13.3%, boundary timeout at 4096) | ⚠️ Partial failure (low retrieval, unverified tokenizer) | **NON-AUTHORITATIVE** | Full Stage 1-4 rerun under dedicated model issue |

---

## 3. Detailed Per-Model Audit Findings

### 3.1 `qwen2.5-0.5b-instruct-q8_0`
- **Artifacts Present:**
  - `results/inventory/qwen2.5-0.5b-instruct-q8_0_Vulkan0_none_full.json`
  - `results/inventory/qwen2.5-0.5b-instruct-q8_0_Vulkan1_none_full.json`
  - `kb/raw/qwen2.5-0.5b-instruct-model-card.md`
  - `kb/wiki/QWEN-001-qwen25-05b-diagnostic.md`
- **Protocol Deficiencies:**
  - *Stage 1:* Deep research was conducted informally without complete Surf CLI verification of Vulkan layer split limits.
  - *Stage 2:* No protocol receipt (`results/receipts/qwen2.5-0.5b-instruct-q8_0.json`) was generated or validated.
  - *Stage 3:* The inventory suite executed single-GPU configurations and attempted tensor split (`-sm tensor`), but did not execute bounded dual-GPU layer split (`-sm layer` with ratios `1,2` / `2,3`). Coarse boundary timed out over SSH at 8192 context.
  - *Stage 4:* The run exhibited low needle-in-a-haystack retrieval (46.7% at 4096 context) and was marked `PARTIAL_FAILURE`. It was never reviewed or signed off as an authoritative benchmark point.
- **Disposition:** **NON-AUTHORITATIVE (Historical Diagnostic Only)**.
- **Next Action:** When scheduled in the model sequence, execute a dedicated issue running the full four-stage protocol with dual-GPU layer split optimization.

---

### 3.2 `Llama-3.2-1B-Instruct-Q4_K_M`
- **Artifacts Present:**
  - `results/inventory/Llama-3.2-1B-Instruct-Q4_K_M_Vulkan0_none_full.json`
  - `results/inventory/Llama-3.2-1B-Instruct-Q4_K_M_Vulkan1_none_full.json`
  - `results/inventory/Llama-3.2-1B-Instruct-Q4_K_M_Vulkan0_Vulkan1_1_1_full_tensor.json`
- **Protocol Deficiencies:**
  - *Stage 1:* No Surf deep research notes exist in `kb/raw/`.
  - *Stage 2:* No wiki entry exists in `kb/wiki/`, and no receipt exists in `results/receipts/`.
  - *Stage 3:* The inventory suite hit boundary SSH timeout at 4096 context. Retrieval rate was severely degraded (13.3% at 2048 context) with multiple remote timeouts. Dual-GPU testing used unsupported tensor split instead of layer split.
  - *Stage 4:* Run was classified as `PARTIAL_FAILURE` due to retrieval collapse and boundary timeouts.
- **Disposition:** **NON-AUTHORITATIVE (Historical Diagnostic Only)**.
- **Next Action:** Re-run through a dedicated issue under the complete four-stage protocol, including tokenizer verification and dual-GPU layer split exploration.

---

## 4. Workload Ordering & Issue Progression

To guarantee rigor, models will be evaluated strictly sequentially:
1. **Issue #24:** Next ordered inventory model investigation (`Qwen3.5-0.8B-Q4_K_M` / smallest available unverified model) through complete four stages.
2. Subsequent model-specific issues created one-by-one.
3. Reruns of `qwen2.5-0.5b-instruct-q8_0` and `Llama-3.2-1B-Instruct-Q4_K_M` under dedicated four-stage issues before closing Issue #1.
4. Final publication of authoritative matrix in `docs/k7000-authoritative-matrix.md` and closure of Issue #1.
