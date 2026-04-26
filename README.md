# FlashKernel-Bench-Custom-CUDA-Triton-Attention-Kernel-Benchmark-Suite
Implement and compare attention-related kernels: baseline PyTorch, a custom CUDA kernel, and possibly Triton variants. Focus on coalesced memory access, shared-memory tiling, async copy, synchronization, and occupancy tradeoffs. Then compare them against vLLM-style attention workloads.
