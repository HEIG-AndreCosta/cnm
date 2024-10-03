#include "matrix.h"
#include <stdlib.h>

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
void tile_square_matrix_multiplication(const int n, const double *matrix_a,
				       const double *matrix_b, double *matrix_c,
				       const int tile_size)
{
	/*
        Implement tile square matrix multiplication!!
        You can use the naive matrix multiplication as base.            
        Iterate over the tiles in each matrix and use tile_multiplication with the correct values        
        Note: Do not forget to handle the edges when the tile is bigger than the matrix.
    */
	double *tile_a =
		(double *)calloc(tile_size * tile_size, sizeof(double));
	double *tile_b =
		(double *)calloc(tile_size * tile_size, sizeof(double));
	double *tile_c =
		(double *)calloc(tile_size * tile_size, sizeof(double));

	const int complete_tiles = n / tile_size;
	const int rest_tile_size = n % tile_size;
}
