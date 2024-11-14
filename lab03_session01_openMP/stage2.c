#include <stdio.h>
#include <omp.h>
int main(void)
{
	int var_1 = 1, var_2 = 2;
#pragma omp parallel private(var_1, var_2)
	{
		int thread_id = omp_get_thread_num();
		int num_threads = omp_get_num_threads();
		printf("Thread %02d of %02d - Vars %i, %i\n", thread_id,
		       num_threads, var_1++, var_2++);
	}
	printf("Vars %i,%i\n", var_1, var_2);
	return 0;
}
