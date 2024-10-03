
#include <assert.h>
#include <stdio.h>
#include "matrix.h"
#include <string.h>

#define N (6)
// Used https://matrix.reshish.com/multCalculation.php to calculate
int main(void)
{
	double a[N][N] = {
		{ 1, 1, 2, 3, 4, 5 },	    { 6, 7, 8, 9, 10, 11 },
		{ 12, 13, 14, 15, 16, 17 }, { 18, 19, 20, 21, 22, 23 },
		{ 24, 25, 26, 27, 28, 29 }, { 30, 31, 32, 33, 34, 35 }
	};
	double b[N][N] = {
		{ 1, 1, 2, 3, 4, 5 },	    { 6, 7, 8, 9, 10, 11 },
		{ 12, 13, 14, 15, 16, 17 }, { 18, 19, 20, 21, 22, 23 },
		{ 24, 25, 26, 27, 28, 29 }, { 30, 31, 32, 33, 34, 35 }
	};
	double target[N][N] = {
		{ 330, 345, 360, 375, 390, 405 },
		{ 870, 921, 972, 1023, 1074, 1125 },
		{ 1410, 1497, 1584, 1671, 1758, 1845 },
		{ 1950, 2073, 2196, 2319, 2442, 2565 },
		{ 2490, 2649, 2808, 2967, 3126, 3285 },
		{ 3030, 3225, 3420, 3615, 3810, 4005 },
	};
	double c[N][N];

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			a[i][j] = i * N + j;
			b[i][j] = i * N + j;
		}
	}

	// naive_matrix_multiplication((double *)a, (double *)b, (double *)c, N, N,
	// 			    N);

	printf("{");
	for (int i = 0; i < N; ++i) {
		printf("{");
		for (int j = 0; j < N; ++j) {
			printf("%g", c[i][j]);
			if (j < N - 1) {
				printf(", ");
			}
		}
		printf("}\n");
	}
	printf("}\n");
	tile_square_matrix_multiplication(N, (double *)a, (double *)b,
					  (double *)c, 2);

	printf("{");
	for (int i = 0; i < N; ++i) {
		printf("{");
		for (int j = 0; j < N; ++j) {
			printf("%g", c[i][j]);
			if (j < N - 1) {
				printf(", ");
			}
		}
		printf("}\n");
	}
	printf("}\n");
	assert(memcmp(target, c, N * N) == 0);
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			assert(target[j][i] == c[j][i]);
		}
	}

	printf("Title Square Matrix Multiplication Tile Size Multiple OK");

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
