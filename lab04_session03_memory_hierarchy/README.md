# Lab 04 - Session 03

## Stage 2

Which statements have coalessed accesss pattern? Explain answers
```c++
__global__ void aux(float *a, int s)
{
    float e = 2.71828;
    int   i = blockDim.x * blockIdx.x + threadIdx.x;    
    ...
```

1. `a[i] = e;` OUI

2. `e = a[i];` OUI

4. `a[i+s] = e;` OUI (dépend de la valeur de `s` s'il en résulte un alignement) 

3. `a[i*s] = e;` OUI si `s == 1` sinon NON


Implement the kernel for number 3 and number 4 and measure the time it takes to complete different executions using different values of `s`. Comment the results.

![alt text](image.png)

Voici le graphe représentant le temps d'exécution en fonction du décalage `s` pour les Kernel Patterns 3 et 4 :

Kernel Pattern 3 : On observe une augmentation progressive du temps d'exécution avec `s` en raison de la `sparsité`.

Kernel Pattern 4 : Le temps reste relativement constant, la dépendence à `s` est moindre.

## Stage 3

**How many times do you read each matrix from global memory in the shared memory implementation?**
La memoire est lue une fois pour chaque tile. Tout dépends de la taille de la tile.

par exemple: pour une matrice A de taille 256 x 256 et une tile de 16 x 16, on lit 16 x 16 = 256 éléments de A. Du coup, pour compléter la matrice A, on lit 256 x 256 / 256 = 256 fois la matrice A.

Si maintenant on optimise au maximum, on peut faire 2 tiles, pour A et B, de 78 x 78. On lit donc 78 x 78 = 6084 éléments de A et B. Pour compléter la matrice A, on lit (256 x 256) / (78 x 78) = 11 fois la matrice A et 11 x la matrice B.

**How long does the gpu gemm with shared memory take?** 

avec `n` = 256 et `m` = 256 et `p` = 256, et des tiles de `[BLOCK_SIZE][BLOCK_SIZE]` avec `BLOCK_SIZE` = 16 on obtient : 

```bash
cnm@cnm-desktop:~/cnm/lab04_session03_memory_hierarchy$ ./gemm
Running GEMM in CPU...
Checking CPU GEMM: Success
Finished GEMM in CPU in 846.309 ms
Running GEMM shared mem in GPU...
Checking GPU shared mem GEMM: Success
Finished GEMM with shared mem in GPU in 4.628 ms
``` 

**Now, double the size of *n*, *m* and *p*. How long does it take now?**

avec `n` = 1024 et `m` = 1024 et `p` = 1024, et des tiles de `[BLOCK_SIZE][BLOCK_SIZE]` avec `BLOCK_SIZE` = 16 on obtient : 

```bash
cnm@cnm-desktop:~/cnm/lab04_session03_memory_hierarchy$ ./gemm
Running GEMM in CPU...
Checking CPU GEMM: Success
Finished GEMM in CPU in 6773.097 ms
Running GEMM shared mem in GPU...
Checking GPU shared mem GEMM: Success
Finished GEMM with shared mem in GPU in 30.849 ms
```

Compare with the results obtained in the previous lab

Tableau comparatif des temps d'exécution pour les différentes tailles de matrices : 
| n x m x p          | CPU (ms) | GPU (ms) | GPU shared mem | `(int) (GPU(ms)/CPU(ms))` | Notes  |
| ------------------ | -------- | -------- | -------------- | ------------------------- | ------ |
| 256 x 256 x 256    | 847.718  | 5.792    | false          | 146 x                     | Lab4.2 |
| 1024 x 1024 x 1024 | 6767.147 | 44.684   | false          | 151 x                     | Lab4.2 |
| 256 x 256 x 256    | 846.309  | 4.628    | true           | 182 x                     | lab4.3 |
| 1024 x 1024 x 1024 | 6773.097 | 30.849   | true           | 219 x                     | lab4.3 |


## Stage 4

**Does the original increment kernel produce the correct result? Why?**
Non, le resultat n'est pas correcte, il y a une race condition en raison de l'accès simultané à la même adresse mémoire par plusieurs threads sans protection/synchronisation.

**How much time does the original increment kernel take?**

Avec `USE_ATOMIC` défini à `0` on obtient : 

```bash
cnm@cnm-desktop:~/cnm/lab04_session03_memory_hierarchy$ nvcc atomic.cu -o at
cnm@cnm-desktop:~/cnm/lab04_session03_memory_hierarchy$ ./at
{ 3748 3748 3748 3748 3748 3748 3748 3748 3748 3748 }
3.19165 ms
```

**How much time does the increment kernel with atomics take? Does it take more or less time than the original kernel? Why?**
Avec `USE_ATOMIC` défini à `1` on obtient : 

```bash
cnm@cnm-desktop:~/cnm/lab04_session03_memory_hierarchy$ nvcc atomic.cu -o at
cnm@cnm-desktop:~/cnm/lab04_session03_memory_hierarchy$ ./at
{ 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000 1000000 }
11.2687 ms
```

On remarque que le temps d'exécution est plus long avec l'utilisation des atomics mais que les races conditions sont évitées.
 
## Stage 5.1
- Performance (speedup) comparison between `simple_gemm` and `cublasSgem`

### Stage 5.1 questions
- What cuBLAS datatype is used for function status returns?
- What does `cublasCreate` do? What data type does is use?
- What cuBLAS methods do we use to copy data from host to GPU? and from GPU to host?
- What matrix memory layout (row-oriented or column-oriented) is assumed in `simple_gemm()`?
- What is the matrix memory layout expected by `cublasSgemm()`?

## Stage 5.2

- What is the network topology?
- What method do we use to create a cuDNN context? What datatype does it use?
- What data type do we use to describe a tensor (a multidimensional array)? What method do we use to create an instance of a tensor descriptor? What method do we use to initalize a multidemsional tensor descriptor?