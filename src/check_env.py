"""Environment / GPU sanity check.

Run this first:  python -m src.check_env

It answers three questions:
1. Is this a CUDA build of PyTorch, and does it see the GPU?
2. How fast is a fp16 matrix multiply compared with fp32?
3. Does a tiny mixed-precision training step actually reduce the loss?
"""

from __future__ import annotations

import platform
import time

import torch


def matmul_benchmark(dtype: torch.dtype, n: int = 4096, iters: int = 20) -> float:
    """Return median-ish seconds per n x n matmul on the GPU."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):  # warm-up (cuBLAS autotuning / clock ramp)
        a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    print(f"platform      : {platform.platform()}")
    print(f"python        : {platform.python_version()}")
    print(f"torch         : {torch.__version__}")
    print(f"CUDA build    : {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible - install the CUDA build of PyTorch.")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM          : {props.total_memory / 1024**3:.2f} GB")
    print(f"capability    : sm_{props.major}{props.minor}")
    print(f"bf16 support  : {torch.cuda.is_bf16_supported()}")

    fp32 = matmul_benchmark(torch.float32)
    fp16 = matmul_benchmark(torch.float16)
    print(f"matmul 4096^3 : fp32 {fp32 * 1e3:.2f} ms | fp16 {fp16 * 1e3:.2f} ms | speedup {fp32 / fp16:.2f}x")

    # --- tiny AMP training step: does the loss go down? ---------------------
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(256, 64, device="cuda")
    y = torch.randint(0, 3, (256,), device="cuda")
    first = last = None
    for step in range(60):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"AMP smoke test: loss {first:.4f} -> {last:.4f} | peak GPU mem {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    assert last < first, "loss did not decrease - something is wrong with the AMP path"
    print("OK: GPU is usable for mixed-precision training.")


if __name__ == "__main__":
    main()
