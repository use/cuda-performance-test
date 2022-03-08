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
    long int *numbers,
    int subBlocks,
    int subTPB
);
__global__ void subKernel(
    long int *numbers,
    int parentBlock,
    int parentThread,
    int parentDim,
    int subBlocks
);

__global__ void subKernel2(
    long int *numbers,
    int parentBlock,
    int parentThread,
    int parentDim,
    int subBlocks
);

void doExperiment(
    int mainBlocks,
    int mainTPB,
    int subBlocks,
    int subTPB
);

int main(int argc, char *argv[])
{
    printf("Main Blocks,Threads Per Block,Sub Blocks,Sub Threads Per Block,Time (ms),Expected Sum, Actual Sum\n");
    int mainBlocks = 16384;
    int mainThreads = 1;
    // int mainBlocks = 32;
    // int mainThreads = 512;
    for (int i = 0; i < 10; i ++) {
        int subBlocks = 1024;
        int subThreads = 1;
        for (int j = 0; j <= 10; j++) {
            doExperiment(mainBlocks, mainThreads, subBlocks, subThreads);
            subBlocks /= 2;
            subThreads *= 2;
        }
        mainBlocks /= 2;
        mainThreads *= 2;
        // mainBlocks *= 2;
        // mainThreads /= 2;
    }
}

void doExperiment(
    int mainBlocks,
    int mainTPB,
    int subBlocks,
    int subTPB
)
{
    int numberQty = mainBlocks * mainTPB * subBlocks * subTPB;
    long int *d_numbers;
    long int *h_numbers = (long int *)malloc(sizeof(long int) * numberQty);
    float msElapsed = 0;
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaMalloc(&d_numbers, sizeof(long int) * numberQty));

    gpuErrchk(cudaEventRecord(start));
    mainKernel<<<mainBlocks, mainTPB>>>(d_numbers, subBlocks, subTPB);
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventElapsedTime(&msElapsed, start, stop));

    gpuErrchk(cudaMemcpy(h_numbers, d_numbers, sizeof(long int) * numberQty, cudaMemcpyDeviceToHost));
    int total = 0;
    for (int i = 0; i < numberQty; i ++)
    {
        total += h_numbers[i];
    }
    printf("%d,%d,%d,%d,%.0f,%d,%d\n",
        mainBlocks, mainTPB, subBlocks, subTPB, msElapsed, numberQty, total);

    cudaFree(d_numbers);
    free(h_numbers);
    cudaDeviceReset();
}

__global__ void mainKernel(
    long int *numbers,
    int subBlocks,
    int subTPB
) {
    subKernel2<<<subBlocks, subTPB>>>(numbers, blockIdx.x, threadIdx.x, blockDim.x, subBlocks);
}

__global__ void subKernel(
    long int *numbers,
    int parentBlock,
    int parentThread,
    int parentDim,
    int subBlocks
) {
    int index = (parentBlock * parentDim + parentThread) * blockDim.x * subBlocks +
        blockDim.x * blockIdx.x + threadIdx.x;
    numbers[index] = 1;
}

__global__ void subKernel2(
    long int *numbers,
    int parentBlock,
    int parentThread,
    int parentDim,
    int subBlocks
) {
    int index = (parentBlock * parentDim + parentThread) * blockDim.x * subBlocks +
        blockDim.x * blockIdx.x + threadIdx.x;
    int total = threadIdx.x;
    while (total < threadIdx.x + 100000) {
        total += 1;
    }
    total = total - threadIdx.x - 100000 + 1;
    numbers[index] = total;
}
