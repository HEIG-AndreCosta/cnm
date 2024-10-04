
#include <assert.h>
#include <stdio.h>
#include "matrix.h"
#include <string.h>

#define N (6)
// Used https://matrix.reshish.com/multCalculation.php to calculate
int main(void)
{
	double a[N][N];
	double b[N][N];

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			a[i][j] = i * N + j;
			b[i][j] = i * N + j;
		}
	}
	double target[N][N] = {
		{ 330, 345, 360, 375, 390, 405 },
		{ 870, 921, 972, 1023, 1074, 1125 },
		{ 1410, 1497, 1584, 1671, 1758, 1845 },
		{ 1950, 2073, 2196, 2319, 2442, 2565 },
		{ 2490, 2649, 2808, 2967, 3126, 3285 },
		{ 3030, 3225, 3420, 3615, 3810, 4005 },
	};
	double c[N][N];

	const int tile_sizes[] = { 1, 2, 3, 6 };
	const size_t array_size = sizeof(tile_sizes) / sizeof(tile_sizes[0]);

	for (int i = 0; i < array_size; ++i) {
		memset(c, 0, N * N);

		tile_square_matrix_multiplication(N, (double *)a, (double *)b,
						  (double *)c, tile_sizes[i]);

		assert(memcmp(target, c, N * N) == 0);

		printf("Tile Matrix Multiplication (N = 6 Tile Size = %d) OK\n",
		       tile_sizes[i]);
	}
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
