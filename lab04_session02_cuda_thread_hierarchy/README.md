# Lab 04 - Session 02

## Stage 2

* Write an example of kernel execution configuration with **block and grid dimensions** bigger that 1. 

```c++
    //Kernel execution configuration example
    kernel<<<1, 1>>>()
```

* How many threads in total will be used with your execution configuration?

## Stage 3

* What happens when use the following execution configuration? Why?
```c++    
    saxpy_gpu<<<4000, 256>>>(n, a, x ,y, z)
    // Where:
    //   n is the vector size
    //   a is the scalar
    //   x,y,z are device vector pointers    
```

* What happens when we use the following execution configuration? Why?

```c++    
    saxpy_gpu<<<1024, 2048>>>(n, a, x ,y, z)
    // Where:
    //   n is the vector size
    //   a is the scalar
    //   x,y,z are device vector pointers    
```

* What happens when we use the following execution configuration? Why?

```c++    
    saxpy_gpu<<<{64,64}, 256>>>(n, a, x ,y, z)
    // Where:
    //   n is the vector size
    //   a is the scalar
    //   x,y,z are device vector pointers    
```
## Stage 4

* How long does the cpu gemm take? and in CUDA? How much is the speed up?

* Now, double the size of *n*, *m* and *p*. How long does the cpu gemm take? and in CUDA? How much is the speed up?