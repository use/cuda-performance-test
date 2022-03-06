#include <stdio.h>

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
    int mainBlocks = 800;
    int mainTPB = 8;
    int subBlocks = 1;
    int subTPB = 200;
    int numberQty = mainBlocks * mainTPB * subBlocks * subTPB;
    int *d_numbers;
    int *h_numbers = (int *)malloc(sizeof(int) * numberQty);

    cudaMalloc(&d_numbers, sizeof(int) * numberQty);
    mainKernel<<<mainBlocks, mainTPB>>>(d_numbers, subBlocks, subTPB);
    cudaMemcpy(h_numbers, d_numbers, sizeof(int) * numberQty, cudaMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < numberQty; i ++)
    {
        total += h_numbers[i];
    }
    printf("Expected: %d\n", numberQty);
    printf("Actual: %d\n", total);
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
