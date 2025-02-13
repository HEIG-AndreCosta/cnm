#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/*
    Tile multiplication as:

        matrix_c = matrix_c + matrix_a * matrix_b

    matrix_a is m by k, column major
    matrix_b is k by n, column major
    matrix_c is m by n, column major
    lda is the leading dimension
*/
void tile_multiplication(const int lda, const double *matrix_a,
			 const double *matrix_b, double *matrix_c, const int M,
			 const int N, const int K)
{
	// row i of matrix_a
	for (int i = 0; i < M; ++i) {
		// column j of matrix_b
		for (int j = 0; j < N; ++j) {
			// calculate cij
			double cij = matrix_c[i + j * lda];
			for (int k = 0; k < K; ++k) {
				cij += matrix_a[i + k * lda] *
				       matrix_b[k + j * lda];
			}
			matrix_c[i + j * lda] = cij;
		}
	}
}
/*
    Matrix tile mutiplication as:
        matrix_c = matrix_c + matrix_a * matrix _b

    matrix a is n by n, column major
    matrix b is n by n, column major
    matrix c is n by n, column major
*/
void tile_square_matrix_multiplication(int n, const double *matrix_a,
				       const double *matrix_b, double *matrix_c,
				       int tile_size)
{
	if (tile_size <= 1) {
		naive_matrix_multiplication(matrix_a, matrix_b, matrix_c, n, n,
					    n);
		return;
	}
	if (tile_size > n) {
		tile_size = n;
	}
	double *tile_a =
		(double *)calloc(tile_size * tile_size, sizeof(double));
	double *tile_b =
		(double *)calloc(tile_size * tile_size, sizeof(double));
	double *tile_c =
		(double *)calloc(tile_size * tile_size, sizeof(double));

	if (!tile_a || !tile_b || !tile_c) {
		free(tile_a);
		free(tile_b);
		free(tile_c);
		return;
	}
	const int complete_tiles = n / tile_size;
	const int nb_tiles = (n * n) / (tile_size * tile_size);

	const int rest_tile_size = n % tile_size;
	if (rest_tile_size == 0) {
		for (int tile = 0; tile < nb_tiles; ++tile) {
			memset(tile_c, 0,
			       tile_size * tile_size * sizeof(*tile_c));
			for (int i = 0; i < complete_tiles; ++i) {
				for (int col = 0; col < tile_size; ++col) {
					const size_t a_index =
						(tile % complete_tiles) *
							tile_size +
						col * n + i * tile_size * n;
					const size_t b_index =
						i * tile_size + col * n +
						((tile / complete_tiles) *
						 tile_size * n);

					memcpy(tile_a + col * tile_size,
					       matrix_a + a_index,
					       tile_size * sizeof(*matrix_a));
					memcpy(tile_b + col * tile_size,
					       matrix_b + b_index,
					       tile_size * sizeof(*matrix_b));
				}
				tile_multiplication(tile_size, tile_a, tile_b,
						    tile_c, tile_size,
						    tile_size, tile_size);
			}
			for (int col = 0; col < tile_size; ++col) {
				const size_t index =
					tile * tile_size + col * n +
					(tile / complete_tiles) * n;

				memcpy(matrix_c + index,
				       tile_c + col * tile_size,
				       tile_size * sizeof(*matrix_c));
			}
		}
	} else {
		// In the case the tile size is not a multiple of the matrix size, we need to check before accessing the matrix values
		//
		// This code is pretty much the same thing as the one above but with more checks
		// The point of this duplication is so we can go faster in the case the tile size is a multiple of the matrix size
		const size_t matrix_size = n * n;
		for (int tile = 0; tile < nb_tiles; ++tile) {
			memset(tile_c, 0,
			       tile_size * tile_size * sizeof(*tile_c));
			for (int i = 0; i < complete_tiles + 1; ++i) {
				for (int col = 0; col < tile_size; ++col) {
					const size_t a_index =
						(tile % complete_tiles) *
							tile_size +
						col * n + i * tile_size * n;
					const size_t b_index =
						i * tile_size + col * n +
						((tile / complete_tiles) *
						 tile_size * n);
					if (a_index < matrix_size) {
						memcpy(tile_a + col * tile_size,
						       matrix_a + a_index,
						       tile_size *
							       sizeof(*matrix_a));
					} else {
						memset(tile_a + col * tile_size,
						       0,
						       tile_size *
							       sizeof(*tile_a));
					}

					if (b_index < matrix_size) {
						memcpy(tile_b + col * tile_size,
						       matrix_b + b_index,
						       tile_size *
							       sizeof(*tile_b));
					} else {
						memset(tile_b + col * tile_size,
						       0,
						       tile_size *
							       sizeof(*tile_b));
					}
				}
				tile_multiplication(tile_size, tile_a, tile_b,
						    tile_c, tile_size,
						    tile_size, tile_size);
			}
			for (int col = 0; col < tile_size; ++col) {
				const size_t index =
					tile * tile_size + col * n +
					(tile / complete_tiles) * n;

				if (index < matrix_size) {
					memcpy(matrix_c + index,
					       tile_c + col * tile_size,
					       tile_size * sizeof(*matrix_c));
				}
			}
		}
	}
	free(tile_a);
	free(tile_b);
	free(tile_c);
}
