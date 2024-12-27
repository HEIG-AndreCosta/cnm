#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <sstream>
#include <iostream>
#include <chrono>

// Includes cuda and cublas
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define FatalError(s)                                                          \
	do {                                                                   \
		std::cout << std::flush << "ERROR: " << s << " in "            \
			  << __FILE__ << ':' << __LINE__ << "\nAborting...\n"; \
		cudaDeviceReset();                                             \
		exit(-1);                                                      \
	} while (0)

#define checkCudaErrors(status)                                                \
	do {                                                                   \
		std::stringstream _err;                                        \
		if (status != 0) {                                             \
			_err << "cuda failure (" << cudaGetErrorString(status) \
			     << ')';                                           \
			FatalError(_err.str());                                \
		}                                                              \
	} while (0)

#define checkCublasErrors(status)                                         \
	do {                                                              \
		std::stringstream _err;                                   \
		if (status != CUBLAS_STATUS_SUCCESS) {                    \
			_err << "cublas failure (code=" << status << ')'; \
			FatalError(_err.str());                           \
		}                                                         \
	} while (0)

/* CPU implementation of a simple version of sgemm */
void simple_sgemm(float *C, const float *A, const float *B, unsigned int n)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j) {
			float sum = 0;
			for (int k = 0; k < n; ++k) {
				sum += A[k * n + i] * B[j * n + k];
			}

			C[j * n + i] = sum;
		}
}

// Check relative error
bool check_sgemm_results(const float *result, const float *reference,
			 unsigned int size)
{
	float error_norm = 0;
	float ref_norm = 0;
	float diff;

	for (int i = 0; i < size; ++i) {
		diff = reference[i] - result[i];
		error_norm += diff * diff;
		ref_norm += reference[i] * reference[i];
	}

	error_norm = (float)sqrt((double)error_norm);
	ref_norm = (float)sqrt((double)ref_norm);
	printf("  %f\n  %f\n", error_norm, ref_norm);
	// Should check ref_norm != 0
	return (error_norm / ref_norm < 1e-6f);
}

int main(int argc, char **argv)
{
	const int N = std::atoi(argv[1]);
	const int M = std::atoi(argv[2]);
	const int P = std::atoi(argv[3]);
	if (!N || !M || !P) {
		printf("N M and P can't be 0\n");
		return 1;
	}
	printf("Running \"%s\" with matrix dim (%i,%i) * (%i,%i) = (%i,%i)\n",
	       argv[0], N, M, N, P, M, P);

	// host memory
	float *h_A, *h_B, *h_C;
	// device memory
	float *d_A, *d_B, *d_C, *h_C_ref;

	float event_elaspsed_time_ms = 0.0f;

	// Allocate host memory for the matrices
	h_A = (float *)malloc(M * N * sizeof(float));
	h_B = (float *)malloc(N * P * sizeof(float));
	h_C = (float *)malloc(M * P * sizeof(float));

	if (h_A == 0 || h_B == 0 || h_C == 0) {
		fprintf(stderr, "Error allocating host memory\n");
		return EXIT_FAILURE;
	}

	// Fill the matrices with test data
	for (size_t i = 0; i < M * N; ++i) {
		h_A[i] = rand() / (float)RAND_MAX;
	}

	for (size_t i = 0; i < N * P; ++i) {
		h_B[i] = rand() / (float)RAND_MAX;
	}

	memset(h_C, 0, M * P * sizeof(float));

	auto cpu_start = std::chrono::high_resolution_clock::now();
	// CPU gemm
	simple_sgemm(h_C, h_A, h_B, N);
	auto cpu_stop = std::chrono::high_resolution_clock::now();

	event_elaspsed_time_ms =
		std::chrono::duration<float, std::milli>(cpu_stop - cpu_start)
			.count();

	printf("\tComplete GEMM in CPU in    %10.3f ms\n",
	       event_elaspsed_time_ms);

	// Save pointer to reference result
	h_C_ref = h_C;

	cublasHandle_t handle;
	checkCublasErrors(cublasCreate(&handle));

	// CUBLAS gemm
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		float total_time = 0.0f;
		unsigned int num_iterations = 10;

		// Allocate device memory for the matrices
		checkCudaErrors(
			cudaMalloc((void **)&d_A, M * N * sizeof(float)));
		checkCudaErrors(
			cudaMalloc((void **)&d_B, N * P * sizeof(float)));
		checkCudaErrors(
			cudaMalloc((void **)&d_C, M * P * sizeof(float)));

		// Initialize the device matrices with the host matrices
		checkCublasErrors(
			cublasSetVector(M * N, sizeof(float), h_A, 1, d_A, 1));
		checkCublasErrors(
			cublasSetVector(N * P, sizeof(float), h_B, 1, d_B, 1));
		checkCublasErrors(
			cublasSetVector(M * P, sizeof(float), h_C, 1, d_C, 1));

		// Warmup operation with cublas
		checkCublasErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
					      M, P, N, &alpha, d_A, M, d_B, N,
					      &beta, d_C, M));

		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		checkCudaErrors(cudaEventRecord(start));

		for (auto j = 0; j < num_iterations; j++) {
			checkCublasErrors(cublasSgemm(
				handle, CUBLAS_OP_N, CUBLAS_OP_N, M, P, N,
				&alpha, d_A, M, d_B, N, &beta, d_C, M));
		}

		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));

		checkCudaErrors(cudaEventElapsedTime(&total_time, start, stop));

		event_elaspsed_time_ms = total_time / num_iterations;
	}

	printf("\tComplete GEMM in CUBLAS in %10.3f ms\n",
	       event_elaspsed_time_ms);

	// Allocate new host memory for CUBLAS result
	h_C = (float *)malloc(M * P * sizeof(float));

	if (h_C == 0) {
		fprintf(stderr, "Error allocating host memory\n");
		return EXIT_FAILURE;
	}

	// Read CUBLAS result
	checkCublasErrors(
		cublasGetVector(M * P, sizeof(float), d_C, 1, h_C, 1));

	bool result_check = check_sgemm_results(h_C, h_C_ref, M * P);

	// Memory clean up
	free(h_A);
	free(h_B);
	free(h_C);
	free(h_C_ref);

	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	checkCublasErrors(cublasDestroy(handle));
	checkCudaErrors(cudaDeviceReset());

	if (result_check) {
		printf("Test PASSED.\n");
		exit(EXIT_SUCCESS);
	} else {
		printf("Test FAILED.\n");
		exit(EXIT_FAILURE);
	}
}
