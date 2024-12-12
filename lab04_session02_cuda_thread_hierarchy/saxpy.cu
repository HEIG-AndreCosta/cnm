// SAXPY Single precision A * X plus Y

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// TODO: CUDA kernel for scalar multiplication and vector addition, pay attention to stride

// Scalar multiplication and vector addition
void saxpy_cpu(int n, float a, float *x, float *y, float *z) {
    for(int i = 0; i < n; ++i){
        z[i] = a * x[i] + y[i];
    }
}

// Check saxpy result
bool check_saxpy(int n, float a, const float *x, const float *y, float *z) {
   for(int i = 0; i < n; ++i) {
        if(z[i] != a * x[i] + y[i]) return false;        
    }
    return true;
}

int main(int argc, char const *argv[])
{
    int n = 1<<20; //2^20
    float  a = 2;

    // Allocate host memory
    float *h_x = (float*)malloc(n*sizeof(float));
    float *h_y = (float*)malloc(n*sizeof(float));
    float *h_z = (float*)malloc(n*sizeof(float));

    if(h_x == NULL || h_y == NULL || h_z == NULL) {
        printf("Error allocating host memory");
        exit(EXIT_FAILURE);
    }   

    // Initialize input vectores
    for(int i = 0; i < n; ++i) {
        h_x[i] = 1.0;
        h_y[i] = (float)(i);
    }

    printf("Running SAXPY in CPU...\n");    
    saxpy_cpu(n, a, h_x, h_y, h_z);
    
    //Check SAXPY CPU results
    printf("Checking CPU SAXPY: %s\n", 
        check_saxpy(n, a, h_x, h_y, h_z)? "Success": "Error");
    
    /* ********************************************************************* */

    //TODO: Device memory allocation and check for errors
    
    //TODO: Copy memory from host to device and check for errors

    //TODO: Call kernel and check for errors

    //TODO: Copy memory from device to host and check for errors
    
    //TODO: Free memory from device to host and check for errors
    
    /* ********************************************************************* */
    
    //Check SAXPY GPU results
    printf("Checking GPU SAXPY: %s\n",
        check_saxpy(n, a, h_x, h_y, h_z)? "Success": "Error");

    free(h_x);
    free(h_y);
    free(h_z);

    return 0;
}
