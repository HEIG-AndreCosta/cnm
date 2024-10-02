#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
    Initilize m by n matrix with random double values between 0 and 1
*/
void matrix_init(double *matrix, const int m, const int n)
{
    for (int i = 0; i < m * n; ++i)
    {
        matrix[i] = drand48();
    }
}

/*
    Clear matrix values 
*/
void matrix_clear(double *matrix, const int m, const int n)
{
    memset(matrix, 0, m * n * sizeof(double));
}

void print_matrix(const double *matrix, const int matrix_size)
{
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            printf("%5.3f ", matrix[j * matrix_size + i]);
        }
        printf("\n");
    }
}