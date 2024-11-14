#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define SIZE 100000000
int main(void)
{
	int *vec_a = (int *)calloc(SIZE, sizeof(int));
	int *vec_b = (int *)calloc(SIZE, sizeof(int));
	int sum, i;
	double start, end;

#pragma omp parallel for
	for (i = 0; i < SIZE; i++) {
		vec_a[i] = 2;
		vec_b[i] = 2;
	}
	sum = 0;
	start = omp_get_wtime();

#pragma omp parallel for
	for (i = 0; i < SIZE; i++) {
		sum += vec_a[i] * vec_b[i];
	}
	end = omp_get_wtime();
	printf("Sum %10i (%.5lfs)\n", sum, end - start);
	return 0;
}
