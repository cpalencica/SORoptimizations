#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 3.0e-9);
}

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


#define PRINT_TIME         1
#define TOL            2e-6
#define OMEGA          1.00
#define ITER           2000

#define IMUL(a, b) __mul24(a, b)

void initializeArray2D(float *arr, int len, int seed);

// kernel for 2a. and 3b.
__global__ void kernel_SOR(int len, float *d_grid, float omega) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    float change;

    // Skip boundary
    if (i == 0 || j == 0 || i == len-1 || j == len-1) return;   
  
    for (int iter = 0; iter < ITER; ++iter) {
        if (i > 0 && i < len-1 && j > 0 && j < len-1) {
            // Compute the new value based on neighbors
            change =
                    d_grid[i * len + j] - 0.25 * (d_grid[(i - 1) * len + j] +
                                                  d_grid[(i + 1) * len + j] +
                                                  d_grid[i * len + j + 1] +
                                                  d_grid[i * len + j - 1]);
                d_grid[i * len + j] -= change * OMEGA;
        }
        __syncthreads();
    }
}

// kernel for 2b.
__global__ void kernel_SOR_multi(int len, float *d_grid, float omega) {
  // get starting index of corresponding thread
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int base_i = tx * 2;
  int base_j = ty * 2;

  for (int iter = 0; iter < ITER; iter++) {
      for (int di = 0; di < 2; di++) {
          for (int dj = 0; dj < 2; dj++) {
              int i = base_i + di;
              int j = base_j + dj;
              // boundary checks
              if (i == 0 || j == 0 || i == len - 1 || j == len - 1)
                  continue;
              float old_val = d_grid[i * len + j];
              float new_val = old_val - omega * (old_val - 0.25f * (
                               d_grid[(i - 1) * len + j] +
                               d_grid[(i + 1) * len + j] +
                               d_grid[i * len + j + 1] +
                               d_grid[i * len + j - 1]
                           ));
              d_grid[i * len + j] = new_val;
          }
      }
      __syncthreads();
  }
}

// kernel for 3a.
__global__ void kernel_block_SOR(int len, float *d_grid, float omega) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i <= 0 || j <= 0 || i >= len-1 || j >= len-1) return;

  float change = d_grid[i * len + j] - 0.25 * (
      d_grid[(i - 1) * len + j] +
      d_grid[(i + 1) * len + j] +
      d_grid[i * len + (j - 1)] +
      d_grid[i * len + (j + 1)]
  );
  d_grid[i * len + j] -= change * omega;
}



int main(int argc, char **argv){
    // size variables
    int blockSizeX = 16;
    int blockSizeY = 16;
    int gridSizeX = 32;
    int gridSizeY = 32;
    int N = blockSizeX * gridSizeX; // Total grid size
  
    // CPU timing variables
    struct timespec time_start, time_stop;

    // GPU Timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;
  
    // Grid on GPU global memory
    float *d_grid;
  
    // Grid on the host memory
    float *h_grid;
    float *h_result;
    float *h_result_gold;

    int errCount = 0, zeroCount = 0;
    float change;

  
    printf("Length of the array = %d\n", N);
  
    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));
  
    // Allocate GPU memory
    size_t allocSize = N * N * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_grid, allocSize));
  
    // Allocate arrays on host memory
    h_grid                     = (float *) malloc(allocSize);
    h_result_gold              = (float *) malloc(allocSize);
    h_result                   = (float *) malloc(allocSize);


    // Initialize the host arrays
    printf("\nInitializing the host grid ...");
    // Arrays are initialized with a known seed for reproducability
    initializeArray2D(h_grid, N, 2453);
    memcpy(h_result_gold, h_grid, allocSize);

    printf("\t... done\n\n");
  
  #if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
  #endif
  
    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_grid, h_grid, allocSize, cudaMemcpyHostToDevice));
  
    // Launch the kernel
    dim3 threadsPerBlock(blockSizeX, blockSizeY);
    dim3 numBlocks(gridSizeX, gridSizeY);
    kernel_SOR<<<numBlocks, threadsPerBlock>>>(N, d_grid, OMEGA);
    
    // Run multiple iterations on host for 3a
    /*
    for (int iter = 0; iter < ITER; ++iter) {
        kernel_block_SOR<<<numBlocks, threadsPerBlock>>>(N, d_grid, OMEGA);
        cudaDeviceSynchronize(); 
    }
        */
  
    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
  
    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_result, d_grid, allocSize, cudaMemcpyDeviceToHost));
  
  #if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  #endif
  
  // compute the results on the host machine
  clock_gettime(CLOCK_REALTIME, &time_start);

  // serial SOR
  for (int iter = 0; iter < ITER; ++iter) {
    for (int i = 1; i < N - 1; ++i) {
      for (int j = 1; j < N - 1; ++j) {
        change =
        h_result_gold[i * N + j] - 0.25 * (h_result_gold[(i - 1) * N + j] +
                                          h_result_gold[(i + 1) * N + j] +
                                          h_result_gold[i * N + j + 1] +
                                          h_result_gold[i * N + j - 1]);
        h_result_gold[i * N + j] -= change * OMEGA;
      }
    }
  }

  

  clock_gettime(CLOCK_REALTIME, &time_stop);
  double time_dur = interval(time_start, time_stop);

  printf("CPU duration: %10.4g (ms)\n", 1000*time_dur);
  
  // === Compare GPU and CPU SOR results ===
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int idx = i * N + j;
      if (fabs(h_result[idx] - h_result_gold[idx]) > TOL) {
        errCount++;
      }
      if (h_result[idx] == 0.0f) {
        zeroCount++;
      }
    }
  }
  
  if (errCount > 0) {
    printf("\n@ERROR: TEST FAILED: %d results did not match\n", errCount);
  } else if (zeroCount > 0) {
    printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
  } else {
    printf("\nTEST PASSED: All results matched\n");
  }

  // Visualize Differences
  /*
  printf("\n--- Full Grid Comparison (CPU vs GPU) ---\n");

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;
            float cpu_val = h_result_gold[idx];
            float gpu_val = h_result[idx];
            float diff = fabs(cpu_val - gpu_val);

            printf("(%2d,%2d): CPU=%7.4f  GPU=%7.4f  Δ=%0.4e", i, j, cpu_val, gpu_val, diff);
            if (diff > TOL) {
                printf("  x");
            } else {
                printf("  o");
            }
            printf("\n");
        }
    }
    */
  
    // Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(d_grid));
  
    free(h_grid);
    free(h_result);
    free(h_result_gold);
  
    return 0;
}

void initializeArray2D(float *arr, int arrLen, int seed) {
    srand(seed);
    for (int i = 0; i < arrLen; i++) {
      for (int j = 0; j < arrLen; j++) {
        arr[i * arrLen + j] = rand() / (float)RAND_MAX;
      }
    }
}