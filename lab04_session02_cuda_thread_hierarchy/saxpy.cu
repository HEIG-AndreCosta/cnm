// SAXPY Single precision A * X plus Y

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// TODO: CUDA kernel for scalar multiplication and vector addition, pay attention to stride

__global__ void scalar_multiplication(size_t n, float a, float *x, float *y,
				      float *z)
{
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < n) {
		z[index] = a * x[index] + y[index];
	}
}
// Scalar multiplication and vector addition
void saxpy_cpu(int n, float a, float *x, float *y, float *z)
{
	for (int i = 0; i < n; ++i) {
		z[i] = a * x[i] + y[i];
	}
}

// Check saxpy result
bool check_saxpy(int n, float a, const float *x, const float *y, float *z)
{
	for (int i = 0; i < n; ++i) {
		if (z[i] != a * x[i] + y[i])
			return false;
	}
	return true;
}

int main(int argc, char const *argv[])
{


    if (argc != 2) {
        printf("Usage: %s <block_size> \n", argv[0]);
        return 1;
    }

    const int block_size = atoi(argv[1]);

    if (block_size <= 0) {
        printf("Block size must be positive integers.\n");
        return 1;
    }

	int n = 1 << 20; //2^20
	float a = 2;

	// Allocate host memory
	float *h_x = (float *)malloc(n * sizeof(float));
	float *h_y = (float *)malloc(n * sizeof(float));
	float *h_z = (float *)malloc(n * sizeof(float));

	if (h_x == NULL || h_y == NULL || h_z == NULL) {
		printf("Error allocating host memory");
		exit(EXIT_FAILURE);
	}

	// Initialize input vectores
	for (int i = 0; i < n; ++i) {
		h_x[i] = 1.0;
		h_y[i] = (float)(i);
	}

	printf("Running SAXPY in CPU...\n");
	saxpy_cpu(n, a, h_x, h_y, h_z);

	//Check SAXPY CPU results
	printf("Checking CPU SAXPY: %s\n",
	       check_saxpy(n, a, h_x, h_y, h_z) ? "Success" : "Error");

	/* ********************************************************************* */

	const size_t size = n * sizeof(float);
	float *d_x, *d_y, *d_z;
	cudaError_t error = cudaMalloc((void **)&d_x, size);
	if (error != cudaSuccess) {
		return 1;
	}
	error = cudaMalloc((void **)&d_y, size);
	if (error != cudaSuccess) {
		cudaFree(d_x);
		return 1;
	}
	error = cudaMalloc((void **)&d_z, size);
	if (error != cudaSuccess) {
		cudaFree(d_x);
		cudaFree(d_y);
		return 1;
	}

	//TODO: Copy memory from host to device and check for errors

	error = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_z);
		return 1;
	}
	error = cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_z);
		return 1;
	}
	error = cudaMemcpy(d_z, h_z, size, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_z);
		return 1;
	}
	//TODO: Call kernel and check for errors

	const size_t num_blocks = (n + block_size - 1) / block_size;

	scalar_multiplication<<<num_blocks, block_size>>>(n, a, d_x, d_y,
							    d_z);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error after kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_z);
        return 1;
    }

	//TODO: Copy memory from device to host and check for errors

	error = cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_z);
		return 1;
	}

	//TODO: Free memory from device to host and check for errors

	error = cudaFree(d_x);
	if (error != cudaSuccess) {
		printf("Couldn't free dx\n");
	}
	error = cudaFree(d_y);
	if (error != cudaSuccess) {
		printf("Couldn't free dy\n");
	}
	error = cudaFree(d_z);
	if (error != cudaSuccess) {
		printf("Couldn't free dz\n");
	}
	/* ********************************************************************* */

	//Check SAXPY GPU results
	printf("Checking GPU SAXPY: %s\n",
	       check_saxpy(n, a, h_x, h_y, h_z) ? "Success" : "Error");

	free(h_x);
	free(h_y);
	free(h_z);

	return 0;
}
