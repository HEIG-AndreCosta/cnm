#include <assert.h>
#include <stdio.h>
#include "matrix.h"
#include <string.h>

#define M	    (3)
#define N	    (4)
#define K	    (2)

// Used https://matrix.reshish.com/multCalculation.php to calculate
int main(void)
{
	double a[K][M] = { { 1, 2, 3 }, { 4, 5, 6 } };
	double b[N][K] = { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } };
	double c[N][M];
	double target[N][M] = {
		{ 9, 12, 15 }, { 19, 26, 33 }, { 29, 40, 51 }, { 39, 54, 69 }
	};

	naive_matrix_multiplication((double *)a, (double *)b, (double *)c, M, N,
				    K);

	assert(memcmp(target, c, M * N) == 0);
	printf("Naive Matrix Multiplication OK\n");
	return 0;

#if 0
	puts("MATRIX A");
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 2; ++j) {
			printf("%g ", a[j][i]);
		}
		printf("\n");
	}
	puts("MATRIX B");
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 4; ++j) {
			printf("%g ", b[j][i]);
		}
		printf("\n");
	}
	puts("MATRIX C");

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 4; ++j) {
			printf("%g ", c[j][i]);
		}
		printf("\n");
	}
#endif
}
