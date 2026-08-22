# Llama-3.2-1B-Instruct-Q4_K_M Research Notes

## Model Metadata
- Checkpoint: meta-llama/Llama-3.2-1B-Instruct
- Quantization: Q4_K_M
- Architecture: LlamaForCausalLM (16 layers, 32 Q heads, 8 KV heads, GQA, intermediate size 8192, vocab size 128256)
- Native Context Window: 131,072 tokens (128k RoPE scaling)
- Size: ~770.3 MiB (807,748,000 bytes)

## Vulkan Runtime Hypotheses
- Single-GPU (Vulkan0/1): Fits comfortably in 2GB VRAM (~800MB weights + KV cache).
- Multi-GPU: Layer split with -sm layer -ts 1,1 (8 layers per GPU) is supported if dual-GPU execution is tested.
