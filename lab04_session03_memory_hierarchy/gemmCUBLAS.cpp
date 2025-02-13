#include <cstdlib>
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
void simple_sgemm(const float *A, const float *B, float *C, unsigned int M,
		  unsigned int N, unsigned int P)
{
	// Iterate over rows in matrix a
	for (size_t i = 0; i < M; ++i) {
		// Iterate over columns in matrix b
		for (size_t j = 0; j < P; ++j) {
			float acc_sum = 0;
			// Iterate over each elment in row and column
			for (size_t k = 0; k < N; ++k) {
				acc_sum += A[i * N + k] * B[k * P + j];
			}
			C[i * P + j] = acc_sum;
		}
	}
}

void squared_sgemm(const float *A, const float *B, float *C, unsigned int n)
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
	double error_norm = 0;
	double ref_norm = 0;
	double diff;

	for (int i = 0; i < size; ++i) {
		diff = reference[i] - result[i];
		error_norm += diff * diff;
		ref_norm += reference[i] * reference[i];
	}

	error_norm = (double)sqrt((double)error_norm);
	ref_norm = (double)sqrt((double)ref_norm);
	printf("  %f\n  %f\n", error_norm, ref_norm);
	// Should check ref_norm != 0
	return (error_norm / ref_norm < 1e-6f);
}

void print_matrix(const float *matrix, unsigned int M, unsigned int N)
{
	for (size_t i = 0; i < M; ++i) {
		for (size_t j = 0; j < N; ++j) {
			printf("%f ", matrix[i * N + j]);
		}
		printf("\n");
	}
}
int main(int argc, char **argv)
{
	if (argc < 4) {
		printf("Usages %s <N> <M> <P>\n", argv[0]);
		return 1;
	}
	const int M = std::atoi(argv[1]);
	const int N = std::atoi(argv[2]);
	const int P = std::atoi(argv[3]);

	const int A_ELEMS = M * N;
	const int B_ELEMS = N * P;
	const int C_ELEMS = M * P;

	if (!N || !M || !P) {
		printf("N M and P can't be 0\n");
		return 1;
	}
	printf("Running \"%s\" with matrix dim (%i,%i) * (%i,%i) = (%i,%i)\n",
	       argv[0], M, N, N, P, M, P);

	// host memory
	float *h_A, *h_B, *h_C;
	// device memory
	float *d_A, *d_B, *d_C, *h_C_ref;

	float event_elaspsed_time_ms = 0.0f;

	// Allocate host memory for the matrices
	h_A = (float *)malloc(A_ELEMS * sizeof(float));
	h_B = (float *)malloc(B_ELEMS * sizeof(float));
	h_C = (float *)calloc(C_ELEMS, sizeof(float));

	if (h_A == 0 || h_B == 0 || h_C == 0) {
		fprintf(stderr, "Error allocating host memory\n");
		return EXIT_FAILURE;
	}

	// Fill the matrices with test data
	for (size_t i = 0; i < A_ELEMS; ++i) {
		h_A[i] = (float)(i + 2) / (i + 1);
	}

	for (size_t i = 0; i < B_ELEMS; ++i) {
		h_B[i] = (float)i / (i + 1);
	}

	auto cpu_start = std::chrono::high_resolution_clock::now();
	// CPU gemm
	simple_sgemm(h_A, h_B, h_C, M, N, P);
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
		unsigned int num_iterations = 1;

		// Allocate device memory for the matrices
		checkCudaErrors(
			cudaMalloc((void **)&d_A, A_ELEMS * sizeof(float)));
		checkCudaErrors(
			cudaMalloc((void **)&d_B, B_ELEMS * sizeof(float)));
		checkCudaErrors(
			cudaMalloc((void **)&d_C, C_ELEMS * sizeof(float)));

		// Initialize the device matrices with the host matrices
		checkCublasErrors(cublasSetVector(A_ELEMS, sizeof(float), h_A,
						  1, d_A, 1));
		checkCublasErrors(cublasSetVector(B_ELEMS, sizeof(float), h_B,
						  1, d_B, 1));

		// Warmup operation with cublas
		checkCublasErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
					      P, M, N, &alpha, d_B, P, d_A, N,
					      &beta, d_C, P));

		cudaEvent_t start, stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		checkCudaErrors(cudaEventRecord(start));
		for (auto j = 0; j < num_iterations; j++) {
			// We trick cublas into calculating or row major order matrices
			// and giving us the result in row major order aswell
			// https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
			checkCublasErrors(cublasSgemm(
				handle, CUBLAS_OP_N, CUBLAS_OP_N, P, M, N,
				&alpha, d_B, P, d_A, N, &beta, d_C, P));
		}

		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));

		checkCudaErrors(cudaEventElapsedTime(&total_time, start, stop));

		event_elaspsed_time_ms = total_time / num_iterations;
	}

	printf("\tComplete GEMM in CUBLAS in %10.3f ms\n",
	       event_elaspsed_time_ms);

	// Allocate new host memory for CUBLAS result
	h_C = (float *)calloc(C_ELEMS, sizeof(float));

	if (h_C == 0) {
		fprintf(stderr, "Error allocating host memory\n");
		return EXIT_FAILURE;
	}

	// Read CUBLAS result
	checkCublasErrors(
		cublasGetVector(C_ELEMS, sizeof(float), d_C, 1, h_C, 1));
	//printf("Matrix A\n");
	//print_matrix(h_A, M, N);
	//printf("Matrix B\n");
	//print_matrix(h_B, N, P);
	//printf("Matrix C GPU\n");
	//print_matrix(h_C, M, P);
	//printf("Matrix C CPU\n");
	//print_matrix(h_C_ref, M, P);

	bool result_check = check_sgemm_results(h_C, h_C_ref, C_ELEMS);

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
