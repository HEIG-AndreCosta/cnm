#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include "matrix.h"

#ifdef BLAS
#include <cblas.h>
#endif

#include "utils.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
    This 3x3 matrix (M)
        | m00 | m01 | m02 |
        | m10 | m11 | m12 |
        | m20 | m21 | m22 |

    In row major would be stored in memory as:
        | m00 | m01 | m02 | m10 | m11 | m12 | m20 | m21 | m22 |
    In column major would be sotre in memory as:
        | m00 | m10 | m20 | m01 | m11 | m21 | m02 | m12 | m22 |

    Element M(1,2) (first row, second column) or m01 in our matrix:
        - In row major would be M[0*3+1] = m01
        - In col major would be M[1*3+0] = m01
    Element M(3,1) (third row, first column) or m20 in our matrix:
        - In row major would be M[2*3+0] = m20
        - In col major would be M[0*3+2] = m20

    If we generalize we have:
        - In row major would be M[row_idx * size + col_idx]
        - In col major would be M[col_idx * size + row_idx]
*/

/*
    Square matrix multiplication as:

        matrix_c = matrix_a * matrix _b

    matrix a is n by n, column major
    matrix b is n by n, column major
    matrix c is n by n, column major
*/
void naive_square_matrix_multiplication(const int n, const double *matrix_a,
					const double *matrix_b,
					double *matrix_c)
{
	// row i of matrix_a
	for (int i = 0; i < n; ++i) {
		// column j of matrix_b
		for (int j = 0; j < n; ++j) {
			// calculate cij
			double cij = 0.0;

			for (int k = 0; k < n; ++k) {
				cij += matrix_a[i + k * n] *
				       matrix_b[k + j * n];
			}
			matrix_c[i + j * n] = cij;
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
}

#ifdef BLAS
/*
    cblas dgemm adaptator
*/
void cblas_square_matrix_multiplication(int n, double *matrix_a,
					double *matrix_b, double *matrix_c)
{
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.,
		    matrix_a, n, matrix_b, n, 1., matrix_c, n);
}
#endif

int main(int argc, char *argv[])
{
	bool use_tiling = false;
	int matrix_size, tile_size, opt;
	char *end_opt_parser;

	if (argc > 1 && argc <= 3) {
		use_tiling = false;
		matrix_size = (int)(strtoul(argv[1], &end_opt_parser, 10));
		if (*end_opt_parser != '\0') {
			fprintf(stderr, "Error parsing matrix_size '%s'",
				argv[1]);
			abort();
		}

		if (argc == 3) {
			use_tiling = true;
			tile_size =
				(int)(strtoul(argv[2], &end_opt_parser, 10));
			if (*end_opt_parser != '\0') {
				fprintf(stderr, "Error parsing tile_size '%s'",
					argv[2]);
				abort();
			}
		}
	} else {
		fprintf(stdout, "Usage %s matrix_size [tile_size]\n", argv[0]);
		abort();
	}

	clock_t start, end;

	double *matrix_a =
		(double *)calloc(matrix_size * matrix_size, sizeof(double));
	double *matrix_b =
		(double *)calloc(matrix_size * matrix_size, sizeof(double));
	double *matrix_c =
		(double *)calloc(matrix_size * matrix_size, sizeof(double));

	if (matrix_a == NULL || matrix_b == NULL || matrix_c == NULL) {
		fprintf(stderr, "Error allocating matrix memory\n");
		abort();
	}

	// "random" seed
	srand48(time(NULL));

	// random matrix initilization
	matrix_init(matrix_a, matrix_size, matrix_size);
	matrix_init(matrix_b, matrix_size, matrix_size);
	matrix_clear(matrix_c, matrix_size, matrix_size);
	// print_matrix(matrix_a, matrix_size);
	// printf("\n");
	// print_matrix(matrix_b, matrix_size);
	// printf("\n");

	/* ----------------------------------------------------------------------------------------- */

	if (use_tiling) {
		// Use tilting matrix multiplication
		start = clock();
		tile_square_matrix_multiplication(
			matrix_size, matrix_a, matrix_b, matrix_c, tile_size);
		end = clock();
	} else {
		// Use naive matrix multiplication
		start = clock();
		naive_matrix_multiplication(matrix_a, matrix_b, matrix_c,
					    matrix_size, matrix_size,
					    matrix_size);
		end = clock();
	}

	printf("%f\n", (float)(end - start) / CLOCKS_PER_SEC);

	/* ----------------------------------------------------------------------------------------- */

	// //Optional cblas section
	// start = clock();
	// cblas_square_matrix_multiplication(matrix_size, matrix_a, matrix_b, matrix_c);
	// end = clock();
	// printf("%f\n", (float)(end - start) / CLOCKS_PER_SEC);

	/* ----------------------------------------------------------------------------------------- */

	free(matrix_a);
	free(matrix_b);
	free(matrix_c);

	return EXIT_SUCCESS;
}
