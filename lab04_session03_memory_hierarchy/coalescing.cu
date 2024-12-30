#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_number_3(float *a, int s)
{
	float e = 2.71828;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i * s] = e;
}

__global__ void kernel_number_4(float *a, int s)
{
	float e = 2.71828;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i + s] = e;
}

int main(int argc, char **argv)
{
	int n = 1 << 20;
	int blockSize = 256;
	float time;
	float *d_a;

	cudaEvent_t startEvent, stopEvent;

	cudaMalloc(&d_a, n * 32 * sizeof(float));

	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	printf("Kernel access pattern number 3:\n");
	for (int i = 0; i <= 32; i++) {
		cudaEventRecord(startEvent, 0);
		kernel_number_3<<<n / blockSize, blockSize>>>(d_a, i);
		cudaEventRecord(stopEvent, 0);

		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&time, startEvent, stopEvent);

		printf("%02d, %.5f ms\n", i, time);
	}

	printf("Kernel access pattern number 4:\n");
	for (int i = 1; i <= 32; i++) {
		cudaEventRecord(startEvent, 0);
		kernel_number_4<<<n / blockSize, blockSize>>>(d_a, i);
		cudaEventRecord(stopEvent, 0);

		cudaEventSynchronize(stopEvent);
		cudaEventElapsedTime(&time, startEvent, stopEvent);

		printf("%02d, %.5f ms\n", i, time);
	}

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
	cudaFree(d_a);
}
