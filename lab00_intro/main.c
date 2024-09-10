#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 10000000

float vec_dot_prod(const float *A, const float *B, const int n){
    float accumulator = 0;
    for(int idx = 0; idx < n; ++idx){
        accumulator = accumulator + (A[idx] * B[idx]);
    }
    return accumulator;
}

void vector_rand_init(float *A, const int n){
    for(int idx = 0; idx < n; ++idx) {
        A[idx] = (float)rand()/(float)(RAND_MAX);
    }
}

int main(int argc, char *argv[]) {
    
    float *vector_a =  (float*)calloc(VECTOR_SIZE, sizeof(float));
    float *vector_b =  (float*)calloc(VECTOR_SIZE, sizeof(float));    
    
    vector_rand_init(vector_a, VECTOR_SIZE);
    vector_rand_init(vector_b, VECTOR_SIZE);
    
    printf("%f\n", vec_dot_prod(vector_a, vector_b, VECTOR_SIZE));

    free(vector_a);
    free(vector_b);
}
