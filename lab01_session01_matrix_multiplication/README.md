
# Lab02 - Matrix Multiplication

Follow the steps of the laboratory and complete the information

## Stage 3 

We are asking you to implement the code for general matrix multiplication (not square).

```C
void naive_matrix_multiplication(const double *matrix_a, const double *matrix_b, double *matrix_c, const int M, const int N, const int K)

```

## Stage 4

Include the measurements and explain your findings based on the analisys of the performance measurements.

## Stage 5

We are asking you to implement the code for tile square matrix multiplication implementing the following function already declared in the code.

```C
void tile_square_matrix_multiplication(const int n, const double *matrix_a, const double *matrix_b, double *matrix_c, const int tile_size)
```

## Stage 6

Include the measurements and explain your findings based on the analysis of the performance measurements.

## Stage 7

Use the perf command to measure the number of cache loads and misses for the general naive and tile implementation.

## Extra

## Notes

You can create a script to take measurements easily. 
This script iterate over a sequence and prints on screen the number and the output of the command.

```sh
    for SIZE in $(seq 100 100 2000)
    do
        echo "$SIZE, $(./main $SIZE)"
    done
```