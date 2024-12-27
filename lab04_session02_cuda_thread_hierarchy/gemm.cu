// SAXPY Single precision A * X plus Y

#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#include <cuda_runtime.h>

#define THRESHOLD 0.0000001

// TODO: CUDA kernel for scalar multiplication and vector addition

__global__ void gemm(float const *a, float const *b, float *c, size_t m,
		     size_t n, size_t p)
{
	size_t row = blockDim.x * blockIdx.x + threadIdx.x;
	size_t col = blockDim.y * blockIdx.y + threadIdx.y;

	printf("Row %zu Col %zu\n", row, col);
	if (row < m && col < p) {
		float acc_sum = 0;
		for (size_t k = 0; k < n; ++k) {
			acc_sum += a[row * n + k] * b[k * p + col];
		}
		c[row * p + col] = acc_sum;
	}
}
// CPU Matrix multiplication
void gemm_cpu(float const *a, float const *b, float *c, size_t m, size_t n,
	      size_t p)
{
	// Iterate over rows in matrix a
	for (size_t i = 0; i < m; ++i) {
		// Iterate over columns in matrix b
		for (size_t j = 0; j < p; ++j) {
			float acc_sum = 0;
			// Iterate over each elment in row and column
			for (size_t k = 0; k < n; ++k) {
				acc_sum += a[i * n + k] * b[k * p + j];
			}
			c[i * p + j] = acc_sum;
		}
	}
}

void print_matrix(float const *a, size_t n)
{
	for (size_t i = 0; i < n; ++i) {
		printf("%g ", a[i]);
	}
	printf("\n");
}
// Check gemm result
bool check_gemm(float const *a, float const *b, float const *c, size_t m,
		size_t n, size_t p)
{
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < p; ++j) {
			float acc_sum = 0;
			for (size_t k = 0; k < n; ++k) {
				acc_sum += a[i * n + k] * b[k * p + j];
			}
			if (abs(c[i * p + j] - acc_sum) > THRESHOLD) {
				return false;
			}
		}
	}
	return true;
}

int main(int argc, char const *argv[])
{
	size_t m = 512;
	size_t n = 512;
	size_t p = 512;

	float event_elaspsed_time_ms = 0;

	// a: m x n
	// b: n x p
	// c: m x p

	// Allocate host memory
	float *h_a = (float *)malloc(m * n * sizeof(float));
	float *h_b = (float *)malloc(n * p * sizeof(float));
	float *h_c = (float *)malloc(m * p * sizeof(float));

	if (h_a == NULL || h_b == NULL || h_c == NULL) {
		printf("Error allocating host memory");
		exit(EXIT_FAILURE);
	}

	// Initialize input matrix
	for (size_t i = 0; i < m * n; ++i) {
		h_a[i] = 1.0;
	};

	for (size_t i = 0; i < n * p; ++i) {
		h_b[i] = 1.0;
	};

	printf("Running GEMM in CPU...\n");
	auto cpu_start = std::chrono::high_resolution_clock::now();
	gemm_cpu(h_a, h_b, h_c, m, n, p);
	auto cpu_stop = std::chrono::high_resolution_clock::now();

	event_elaspsed_time_ms =
		std::chrono::duration<float, std::milli>(cpu_stop - cpu_start)
			.count();

	printf("Complete GEMM in CPU in %.3f ms\n", event_elaspsed_time_ms);

	// Check SAXPY CPU results
	printf("Checking CPU GEMM: %s\n",
	       check_gemm(h_a, h_b, h_c, m, n, p) ? "Success" : "Error");

	print_matrix(h_c, m * p);

	// Clean up result
	for (size_t i = 0; i < m * p; ++i) {
		h_c[i] = 0.0;
	};

	/* ********************************************************************* */

	// TODO: device memory allocation

	const size_t size_a = m * n * sizeof(float);
	const size_t size_b = n * p * sizeof(float);
	const size_t size_c = m * p * sizeof(float);

	float *d_a, *d_b, *d_c;
	cudaError_t error = cudaMalloc((void **)&d_a, size_a);
	if (error != cudaSuccess) {
		return 1;
	}
	error = cudaMalloc((void **)&d_b, size_b);
	if (error != cudaSuccess) {
		cudaFree(d_a);
		return 1;
	}
	error = cudaMalloc((void **)&d_c, size_c);
	if (error != cudaSuccess) {
		cudaFree(d_a);
		cudaFree(d_b);
		return 1;
	}

	//TODO: Copy memory from host to device and check for errors

	error = cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return 1;
	}
	error = cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return 1;
	}
	error = cudaMemcpy(d_c, h_c, size_c, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return 1;
	}
	//TODO: Call kernel and check for errors

	const size_t block_size = 256;
	const size_t num_blocks = ((m * n) + block_size - 1) / block_size;

	gemm<<<num_blocks, block_size>>>(d_a, d_b, d_c, m, n, p);

	//TODO: Copy memory from device to host and check for errors

	error = cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		return 1;
	}
	print_matrix(h_c, m * p);

	//TODO: Free memory from device to host and check for errors

	error = cudaFree(d_a);
	if (error != cudaSuccess) {
		printf("Couldn't free dx\n");
	}
	error = cudaFree(d_b);
	if (error != cudaSuccess) {
		printf("Couldn't free dy\n");
	}
	error = cudaFree(d_c);
	if (error != cudaSuccess) {
		printf("Couldn't free dz\n");
	}

	/* ********************************************************************* */

	// Check SAXPY GPU results
	printf("Checking GPU GEMM: %s\n",
	       check_gemm(h_a, h_b, h_c, m, n, p) ? "Success" : "Error");

	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
