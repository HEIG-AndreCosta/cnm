#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_THREADS 10000000
#define BLOCK_WIDTH 1000

__global__ void increment(int *a, size_t size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	i = i % size;
	atomicInc(a + i, 1);
}

// TODO implement increment kernel with CUDA atomics

int main(int argc, char const *argv[])
{
	size_t size = 10;

	int *h_a = new int[size];
	int *d_a;

	float event_elaspsed_time_ms = 0.0;

	/* ********************************************************************* */

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void **)&d_a, size * sizeof(int));
	cudaMemset((void *)d_a, 0, size * sizeof(int));

	cudaEventRecord(start);
	increment<<<std::ceil(NUM_THREADS / BLOCK_WIDTH), BLOCK_WIDTH>>>(
		d_a, size);
	cudaEventRecord(stop);

	cudaMemcpy(h_a, d_a, size * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventElapsedTime(&event_elaspsed_time_ms, start, stop);

	cudaFree(d_a);

	/* ********************************************************************* */

	printf("{");
	for (size_t i = 0; i < size; ++i)
		printf(" %d", h_a[i]);
	printf(" }\n");

	printf("%g ms\n", event_elaspsed_time_ms);

	delete[] h_a;

	return 0;
}
