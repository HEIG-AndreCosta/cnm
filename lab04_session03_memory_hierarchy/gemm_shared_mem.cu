#include <chrono>
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define THRESHOLD  0.00001

// a: m x n
// b: n x p
// c: m x p
__global__ void gemm_shared_mem(float *a, float *b, float *c, size_t m,
				size_t n, size_t p)
{
	// row a column based on thread index inside the block and grid
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0;

	const int TILE_SIZE = blockDim.x;

	// Shared memory for tiles
	__shared__ float aTile
		[BLOCK_SIZE]
		[BLOCK_SIZE]; // TILE_SIZE must not exceed shared memory limits
	__shared__ float bTile[BLOCK_SIZE][BLOCK_SIZE];

	// Iterate over tiles in the shared dimension
	for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
		// Load elements of the current tile from global memory into shared memory
		if (row < m && t * TILE_SIZE + threadIdx.x < n) {
			aTile[threadIdx.y][threadIdx.x] =
				a[row * n + t * TILE_SIZE + threadIdx.x];
		} else {
			aTile[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if (col < p && t * TILE_SIZE + threadIdx.y < n) {
			bTile[threadIdx.y][threadIdx.x] =
				b[(t * TILE_SIZE + threadIdx.y) * p + col];
		} else {
			bTile[threadIdx.y][threadIdx.x] = 0.0f;
		}
		// Synchronize to ensure all threads have loaded their tiles
		__syncthreads();

		// Multiply the tiles
		for (int k = 0; k < TILE_SIZE; ++k) {
			sum += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
		}
		// Synchronize to ensure the tiles aren't overwritten
		__syncthreads();
	}

	c[row * m + col] += sum;
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
	size_t m = 1024;
	size_t n = 1024;
	size_t p = 1024;

	float event_elaspsed_time_ms = 0;

	// a: m x n
	// b: n x p
	// c: m x p

	// Allocate host memory
	float *h_a = new float[m * n];
	float *h_b = new float[n * p];
	float *h_c = new float[m * p];

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

	// Check SAXPY CPU results
	printf("Checking CPU GEMM: %s\n",
	       check_gemm(h_a, h_b, h_c, m, n, p) ? "Success" : "Error");

	event_elaspsed_time_ms =
		std::chrono::duration<float, std::milli>(cpu_stop - cpu_start)
			.count();

	printf("Finished GEMM in CPU in %.3f ms\n", event_elaspsed_time_ms);

	// Clean up result
	for (size_t i = 0; i < m * p; ++i) {
		h_c[i] = 0.0;
	};

	/* ********************************************************************* */

	float *d_a, *d_b, *d_c;

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Device memory allocation
	cudaMalloc((void **)&d_a, m * n * sizeof(float));
	cudaMalloc((void **)&d_b, n * p * sizeof(float));
	cudaMalloc((void **)&d_c, m * p * sizeof(float));

	// Device memory inialization
	cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n * p * sizeof(float), cudaMemcpyHostToDevice);

	// Excution configuration
	dim3 grid_size(m / BLOCK_SIZE, p / BLOCK_SIZE);
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

	// Kernel execution and measuring
	printf("Running GEMM shared mem in GPU...\n");
	cudaEventRecord(start);
	gemm_shared_mem<<<grid_size, block_size> > >(d_a, d_b, d_c, m, n, p);
	cudaEventRecord(stop);

	// Copy result from device
	cudaMemcpy(h_c, d_c, m * p * sizeof(float), cudaMemcpyDeviceToHost);

	// Check result
	printf("Checking GPU shared mem GEMM: %s\n",
	       check_gemm(h_a, h_b, h_c, m, n, p) ? "Success" : "Error");

	cudaEventElapsedTime(&event_elaspsed_time_ms, start, stop);

	printf("Finished GEMM with shared mem in GPU in %.3f ms\n",
	       event_elaspsed_time_ms);

	// Event cleaning
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Memory deallocation
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] h_a;
	delete[] h_b;
	delete[] h_c;

	return 0;
}
