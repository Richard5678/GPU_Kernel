import torch
import triton
import triton.language as tl
import time


@triton.jit
def matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    # Get thread indices exactly like CUDA
    # int r = blockIdx.y * blockDim.y + threadIdx.y;
    # int c = blockIdx.x * blockDim.x + threadIdx.x;
    r = tl.program_id(0)
    c = tl.program_id(1)

    # Only compute if within bounds (exactly like CUDA)
    if r < M and c < N:
        # Initialize accumulator for single element
        acc = 0.0

        # Compute exactly like CUDA's loop
        for k in range(K):
            # Same indexing as CUDA: A[r * s + i] and B[c + i * n]
            a = tl.load(A + r * stride_am + k * stride_ak)
            b = tl.load(B + k * stride_bk + c * stride_bn)
            acc += a * b

        # Store single element result
        tl.store(C + r * stride_cm + c * stride_cn, acc)


def matmul(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Use exact same grid/block dimensions as CUDA
    BLOCK_SIZE = 32
    grid = (M, N)  # Each thread computes one element

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return c


if __name__ == "__main__":
    M, N, K = 1000, 700, 500  # Same dimensions as CUDA example
    a = torch.randn((M, K), device="cuda")
    b = torch.randn((K, N), device="cuda")

    # Warmup
    matmul(a, b)
    torch.cuda.synchronize()

    # Time it
    start = time.perf_counter()
    c = matmul(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"Triton matmul took {(end - start) * 1000:.3f} ms")
