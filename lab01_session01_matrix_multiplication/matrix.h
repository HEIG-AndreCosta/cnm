#ifndef MATRIX_H
#define MATRIX_H

void naive_matrix_multiplication(const double *matrix_a, const double *matrix_b,
				 double *matrix_c, const int M, const int N,
				 const int K);

void tile_square_matrix_multiplication(int n, const double *matrix_a,
				       const double *matrix_b, double *matrix_c,
				       int tile_size);

void tile_multiplication(const int lda, const double *matrix_a,
			 const double *matrix_b, double *matrix_c, const int M,
			 const int N, const int K);
#endif
