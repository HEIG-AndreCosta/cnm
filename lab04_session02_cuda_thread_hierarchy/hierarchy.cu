#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void print_variables(void)
{

	printf("Thread (%d, %d)/(%d, %d) in (%d, %d)/(%d, %d)\n", threadIdx.x,
	       threadIdx.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y,
	       gridDim.x, gridDim.y);
}

int main(int argc, char const *argv[])
{
	dim3 blockDim(6, 6); // Block size of 16x16 threads
	dim3 gridDim(5, 5);
	print_variables<<<blockDim, gridDim>>>();
	cudaDeviceSynchronize();
	return 0;
}
