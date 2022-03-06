#include <stdio.h>
#include <sys/time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void mainKernel(
    int *numbers,
    int subBlocks,
    int subTPB
);
__global__ void subKernel(
    int *numbers,
    int parentBlock,
    int parentThread,
    int parentDim,
    int subBlocks
);

int main(int argc, char *argv[])
{
    // int mainBlocks = 800;
    // int mainTPB = 8;
    int mainBlocks = 6400;
    int mainTPB = 1;
    int subBlocks = 16;
    int subTPB = 16;
    int numberQty = mainBlocks * mainTPB * subBlocks * subTPB;
    int *d_numbers;
    int *h_numbers = (int *)malloc(sizeof(int) * numberQty);
    float msElapsed = 0;
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaMalloc(&d_numbers, sizeof(int) * numberQty));

    gpuErrchk(cudaEventRecord(start));
    mainKernel<<<mainBlocks, mainTPB>>>(d_numbers, subBlocks, subTPB);
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventElapsedTime(&msElapsed, start, stop));

    gpuErrchk(cudaMemcpy(h_numbers, d_numbers, sizeof(int) * numberQty, cudaMemcpyDeviceToHost));
    int total = 0;
    for (int i = 0; i < numberQty; i ++)
    {
        total += h_numbers[i];
    }
    printf("Expected: %d\n", numberQty);
    printf("Actual:   %d\n", total);
    printf("Kernel Time (ms): %.4f\n", msElapsed);
}

__global__ void mainKernel(
    int *numbers,
    int subBlocks,
    int subTPB
) {
    subKernel<<<subBlocks, subTPB>>>(numbers, blockIdx.x, threadIdx.x, blockDim.x, subBlocks);
}

__global__ void subKernel(
    int *numbers,
    int parentBlock,
    int parentThread,
    int parentDim,
    int subBlocks
) {
    int index = (parentBlock * parentDim + parentThread) * blockDim.x * subBlocks +
        blockDim.x * blockIdx.x + threadIdx.x;
    numbers[index] = 1;
}
