# Lab 04 - Session 02

## Stage 2

* Write an example of kernel execution configuration with **block and grid dimensions** bigger that 1. 

```c++
    //Kernel execution configuration example
	dim3 blockDim(6, 6);
	dim3 gridDim(5, 5);
	print_variables<<<blockDim, gridDim>>>();

```

Voici une partie de l'output:

```bash
Thread (0, 0)/(5, 5) in (4, 5)/(6, 6)
Thread (1, 0)/(5, 5) in (4, 5)/(6, 6)
Thread (2, 0)/(5, 5) in (4, 5)/(6, 6)
Thread (3, 0)/(5, 5) in (4, 5)/(6, 6)
Thread (4, 0)/(5, 5) in (4, 5)/(6, 6)
Thread (0, 1)/(5, 5) in (4, 5)/(6, 6)
Thread (1, 1)/(5, 5) in (4, 5)/(6, 6)
Thread (2, 1)/(5, 5) in (4, 5)/(6, 6)
Thread (3, 1)/(5, 5) in (4, 5)/(6, 6)
Thread (4, 1)/(5, 5) in (4, 5)/(6, 6)
Thread (0, 2)/(5, 5) in (4, 5)/(6, 6)
Thread (1, 2)/(5, 5) in (4, 5)/(6, 6)
Thread (2, 2)/(5, 5) in (4, 5)/(6, 6)
Thread (3, 2)/(5, 5) in (4, 5)/(6, 6)
Thread (4, 2)/(5, 5) in (4, 5)/(6, 6)
Thread (0, 3)/(5, 5) in (4, 5)/(6, 6)
Thread (1, 3)/(5, 5) in (4, 5)/(6, 6)
Thread (2, 3)/(5, 5) in (4, 5)/(6, 6)
Thread (3, 3)/(5, 5) in (4, 5)/(6, 6)
Thread (4, 3)/(5, 5) in (4, 5)/(6, 6)
Thread (0, 4)/(5, 5) in (4, 5)/(6, 6)
...
Thread (1, 4)/(5, 5) in (4, 5)/(6, 6)
Thread (2, 4)/(5, 5) in (4, 5)/(6, 6)
Thread (3, 4)/(5, 5) in (4, 5)/(6, 6)
Thread (4, 4)/(5, 5) in (4, 5)/(6, 6)
Thread (0, 0)/(5, 5) in (3, 5)/(6, 6)
Thread (1, 0)/(5, 5) in (3, 5)/(6, 6)
Thread (2, 0)/(5, 5) in (3, 5)/(6, 6)
Thread (3, 0)/(5, 5) in (3, 5)/(6, 6)
Thread (4, 0)/(5, 5) in (3, 5)/(6, 6)
Thread (0, 1)/(5, 5) in (3, 5)/(6, 6)
Thread (1, 1)/(5, 5) in (3, 5)/(6, 6)
Thread (2, 1)/(5, 5) in (3, 5)/(6, 6)
Thread (3, 1)/(5, 5) in (3, 5)/(6, 6)
Thread (4, 1)/(5, 5) in (3, 5)/(6, 6)
Thread (0, 2)/(5, 5) in (3, 5)/(6, 6)
Thread (1, 2)/(5, 5) in (3, 5)/(6, 6)
Thread (2, 2)/(5, 5) in (3, 5)/(6, 6)
Thread (3, 2)/(5, 5) in (3, 5)/(6, 6)
Thread (4, 2)/(5, 5) in (3, 5)/(6, 6)
Thread (0, 3)/(5, 5) in (3, 5)/(6, 6)
Thread (1, 3)/(5, 5) in (3, 5)/(6, 6)
Thread (2, 3)/(5, 5) in (3, 5)/(6, 6)
Thread (3, 3)/(5, 5) in (3, 5)/(6, 6)
Thread (4, 3)/(5, 5) in (3, 5)/(6, 6)
Thread (0, 4)/(5, 5) in (3, 5)/(6, 6)
Thread (1, 4)/(5, 5) in (3, 5)/(6, 6)
Thread (2, 4)/(5, 5) in (3, 5)/(6, 6)
Thread (3, 4)/(5, 5) in (3, 5)/(6, 6)
Thread (4, 4)/(5, 5) in (3, 5)/(6, 6)
``` 


**How many threads in total will be used with your execution configuration?**

32 x 32 = 1024

## Stage 3

** What happens when use the following execution configuration? Why?**

```c++    
    scalar_multiplication<<<4000, 256>>>(n, a, x ,y, z)
    // Where:
    //   n is the vector size
    //   a is the scalar
    //   x,y,z are device vector pointers    
```

Le programme se lance avec succés car le nombre de blocs est inférieur à 65536 et que le nombre de threads par bloc est inférieur à 1024.

**What happens when we use the following execution configuration? Why?**

```c++    
    scalar_multiplication<<<1024, 2048>>>(n, a, x ,y, z)
    // Where:
    //   n is the vector size
    //   a is the scalar
    //   x,y,z are device vector pointers    
```

Ici ca ne fonctionne pas car le nombre de threads par bloc est supérieur à 1024. 

**What happens when we use the following execution configuration? Why?**

```c++    
    scalar_multiplication<<<{64,64}, 256>>>(n, a, x ,y, z)
    // Where:
    //   n is the vector size
    //   a is the scalar
    //   x,y,z are device vector pointers    
```

Dans ce cas, nous avons une grille de 64 x 64 ce qui nous donne un total de 4096 blocs. Cela fonctionne pour les memes raisons que le premier exemple.

## Stage 4

* How long does the cpu gemm take? and in CUDA? How much is the speed up?

avec `n = 512`, `m = 512` et `p = 512`:

```bash
cnm@cnm-desktop:~/cnm/lab04_session02_cuda_thread_hierarchy$ ./gemm 
Running GEMM in CPU...
Complete GEMM in CPU in 847.718 ms
Checking CPU GEMM: Success
Complete GEMM in GPU in 5.792 ms (with cuda event)
Checking GPU GEMM: Success
``` 

Le calcul sur GPU est appriximativement `146` fois plus rapide que sur CPU.

* Now, double the size of *n*, *m* and *p*. How long does the cpu gemm take? and in CUDA? How much is the speed up?

avec `n = 1024`, `m = 1024` et `p = 1024`:

```bash
cnm@cnm-desktop:~/cnm/lab04_session02_cuda_thread_hierarchy$ ./gemm 
Running GEMM in CPU...
Complete GEMM in CPU in 6767.147 ms
Checking CPU GEMM: Success
Complete GEMM in GPU in 44.684 ms (with cuda event)
Checking GPU GEMM: Success
```

Le calcul sur GPU est appriximativement `151` fois plus rapide que sur CPU.
