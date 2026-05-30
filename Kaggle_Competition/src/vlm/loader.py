"""
VLM Loader — GPU-Optimized for L40s (48 GB VRAM)
Model: Qwen2.5-VL-72B-Instruct (NF4 4-bit ≈ 38-42 GB)

Optimizations applied:
  - Flash Attention 2 (Ada Lovelace / L40s native)
  - NF4 double quantization (BitsAndBytes)
  - bfloat16 compute dtype (L40s BF16 tensor cores)
  - Single CUDA device, no CPU offload
  - TF32 enabled for GEMMs
  - KV cache always on
"""
import torch
import gc
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

# ── Global singleton references ──────────────────────────────────────────────
_processor: AutoProcessor | None = None
_model: Qwen2_5_VLForConditionalGeneration | None = None


def _configure_gpu() -> None:
    """Apply global CUDA performance flags for L40s."""
    # Allow TF32 matmuls on Ampere/Ada (L40s has Ada Lovelace arch)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # More aggressive CUDA memory allocator
    torch.cuda.set_per_process_memory_fraction(0.95)


def get_vlm(model_path: str, use_4bit: bool = True):
    """
    Load Qwen2.5-VL-72B-Instruct on the L40s.
    Returns cached (processor, model) if already loaded.

    Args:
        model_path: Local directory with downloaded model weights.
        use_4bit:   NF4 4-bit quantization (required for 72B on 48 GB).

    Returns:
        (processor, model) tuple both on CUDA.
    """
    global _processor, _model

    if _model is not None and _processor is not None:
        return _processor, _model

    _configure_gpu()
    print(f"[Loader] Loading VLM from: {model_path}")
    print(f"[Loader] CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"[Loader] Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Quantization config (NF4 double-quant) ────────────────────────────
    quant_cfg = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",               # Normal Float 4 (best quality)
            bnb_4bit_compute_dtype=torch.bfloat16,   # BF16 compute on L40s tensor cores
            bnb_4bit_use_double_quant=True,           # Double quantization saves ~0.5 GB
            # Keep quantized weights on GPU (no CPU offload)
            llm_int8_enable_fp32_cpu_offload=False,
        )

    # ── Processor ─────────────────────────────────────────────────────────
    _processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        # Use maximum pixels for high-quality image understanding
        max_pixels=1280 * 28 * 28,
        min_pixels=256 * 28 * 28,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,      # BF16 weights/activations
        device_map="cuda:0",             # Single-GPU, no CPU offload
        attn_implementation="sdpa",  # SDPA instead of Flash Attn 2
        local_files_only=True,
    )
    _model.eval()  # Disable dropout, etc.

    # Log memory usage after loading
    mem_allocated = torch.cuda.memory_allocated(0) / 1e9
    mem_reserved  = torch.cuda.memory_reserved(0) / 1e9
    print(f"[Loader] Model loaded. VRAM allocated: {mem_allocated:.2f} GB | reserved: {mem_reserved:.2f} GB")

    return _processor, _model


def free_memory() -> None:
    """Release model from GPU memory."""
    global _processor, _model
    _processor = None
    _model = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[Loader] GPU memory freed.")
