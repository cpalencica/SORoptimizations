# Successive Over-Relaxation (SOR) Parallel Optimization Project
Authors: Cristian Palencia and Yiran Yin

## Overview

This project explores various parallelization strategies to accelerate the Successive Over-Relaxation (SOR) method for solving a 2D Laplace equation on a grid. It includes implementations using:

- **Pthreads** — explicit thread management on CPU,
- **OpenMP** — compiler-driven multi-threading on CPU,
- **CUDA** — GPU parallelization with several kernel variants.

The goal is to compare performance, scalability, and ease of implementation across CPU and GPU parallel models, while maintaining result correctness.

## Implementations

### 1. Pthreads SOR (`test_sor_mt.c`)

- Manually creates and manages POSIX threads.
- Divides the grid among threads for parallel updates.
- Synchronizes threads to ensure proper iteration steps.
- Demonstrates low-level control over CPU parallelism.
- Useful for understanding thread coordination and synchronization primitives.
- Also tested with different loop orderings

### 2. OpenMP SOR (`test_sor_omp.c`)

- Uses OpenMP pragmas to parallelize nested loops on CPU.
- Compiler handles thread creation, work distribution, and synchronization.
- Simpler to implement compared to Pthreads.
- Easily scalable with minimal code changes.
- Good for rapid prototyping and testing CPU parallelism.
- Also applied to red-black, loop reordering, and blocked implementations

### 3. CUDA SOR (`cuda_sor.cu`)

- Offloads computation to NVIDIA GPUs.
- Includes multiple CUDA kernel variants:
  - Standard kernel: One thread per grid point, iterating inside the kernel.
  - Multi-point kernel: Each thread updates a 2x2 block to improve memory throughput.
  - Blocked kernel: Single iteration per kernel launch, with multiple launches to allow fine-grained synchronization.
- Uses CUDA events and CPU timers to measure GPU and CPU execution times.
- Compares GPU results with CPU baseline to ensure accuracy.

## Project Structure

| File              | Description                                |
|-------------------|--------------------------------------------|
| `test_sor_mt.c`   | Pthreads implementation of SOR             |
| `test_sor_omp.c`  | OpenMP implementation of SOR                |
| `cuda_sor.cu`     | CUDA implementation with multiple kernel variants |
| `README.md`       | Project overview and usage instructions     |

## Performance Measurement

- Each version times the execution of a fixed number of SOR iterations (`ITER`).
- CPU versions use `clock_gettime`.
- CUDA uses CUDA events for GPU timing.
- Results from all implementations are compared to ensure numerical correctness within a small tolerance (`TOL`).

## Notes and Extensions

- **Pthreads:** Good for learning thread management and synchronization; more boilerplate code required.
- **OpenMP:** Easier to maintain and scale; ideal for CPU-bound parallel tasks.
- **CUDA:** Offers highest parallelism and speedup on suitable hardware; requires knowledge of GPU programming concepts.


