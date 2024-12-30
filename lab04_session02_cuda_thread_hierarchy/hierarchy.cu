#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void print_variables(void)
{
	printf("Thread %d/%d in %d/%d\n", threadIdx.x, blockDim.x, blockIdx.x,
	       gridDim.x);
}

int main(int argc, char const *argv[])
{
	print_variables<<<32, 32>>>();
	cudaDeviceSynchronize();
	return 0;
}
