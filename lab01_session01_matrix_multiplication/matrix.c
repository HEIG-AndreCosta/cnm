#include "matrix.h"

/*
    Matrix multiplication as:

        matrix_c = matrix_a * matrix_b

    matrix_a is M by K, column major
    matrix_b is K by N, column major
    matrix_c is M by N, column major
*/
void naive_matrix_multiplication(const double *matrix_a, const double *matrix_b,
				 double *matrix_c, const int M, const int N,
				 const int K)
{
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			double cmn = 0;
			for (int k = 0; k < K; ++k) {
				double ai = matrix_a[k * M + i];
				double bj = matrix_b[j * K + k];
				cmn += ai * bj;
			}
			matrix_c[j * M + i] = cmn;
		}
	}
}
