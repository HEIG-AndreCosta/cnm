<style>
  legends {
    font-size: 0.8em;
    color: #333;
  }
</style>

<img src="heig-vd.png" style="height:80px;">

# <center> Session n°1 {ignore=true}

# <center> Matrix multiplication optimization {ignore=true}

## <center>Departement : TIC {ignore=true}

## <center>CNM{ignore=true}

Author: **Andre Costa & Alexandre Iorio**

Professor: **Marina Zapater**

Assistant : **Mehdi Akeddar**

Classroom : **B03**

Date : **05.10.2024**

<!-- pagebreak -->

<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=4 orderedList=false} -->

## Table des matières {ignore=true}

<!-- code_chunk_output -->

- [0. Preface](#0-preface)
- [1. Introduction](#1-introduction)
- [2. Material](#2-material)
- [3. Stage 1 - Understanding matrix memory layout and matrix multiplication](#3-stage-1---understanding-matrix-memory-layout-and-matrix-multiplication)
  - [3.1. Memory layout](#31-memory-layout)
  - [3.2. Matrix multiplication algorithm](#32-matrix-multiplication-algorithm)
- [4. Stage 2 - Source template and compiler flags](#4-stage-2---source-template-and-compiler-flags)
- [5. Stage 3 - Implementing general matrix multiplication](#5-stage-3---implementing-general-matrix-multiplication)
- [6. Stage 4 - Measuring naïve matrix multiplication performance](#6-stage-4---measuring-naïve-matrix-multiplication-performance)
- [9. Conclusion](#9-conclusion)
- [10. Ref](#10-ref)

<!-- /code_chunk_output -->

<!-- pagebreak -->

## 0. Preface

This lab is written in Markdown format optimized for the interpreter used by the `Markdown Preview Enhanced` plugin of `Visual Studio Code`.

## 1. Introduction

The objective of this session is to explore and optimize the matrix multiplication algorithm. 

We will analyze the cache memory behavior of the algorithm and optimize it using the tiling technique.

## 2. Material

To realize this laboratory, we will use the an `Nvidia® Jetson Orin Nano`.

To realize measurements, we will use the return of the `main` function and the `perf` command to measure the number of cache loads and misses for the general naive and tile implementation. To simplify the process, we will use a script to take measurements easily. 

If you want to use the script, be sure taht you have the `perf` command installed on your system this `python3` libraries:
- `matplotlib`
- `tqdm` (To show progress bar and to be sure that the script is running)

Here is the usage of `perf.py` script:

```sh
python3 perf.py -h
usage: perf.py [-h] -s START -e END -i INCREMENT [-t TILE] [-c] [-T]

Matrix multiplication performance script.

options:
  -h, --help            show this help message and exit
  -s START, --start START
                        Start matrix size.
  -e END, --end END     End matrix size.
  -i INCREMENT, --increment INCREMENT
                        Increment for matrix sizes.
  -t TILE, --tile TILE  Tile size for tiled multiplication.
  -c, --cache           Measure cache usage.
  -T, --time            Measure execution time.
  -S, --save            Save the plot as an SVG file.
  -F, --file            Name of the saved file.
``` 

## 3. Stage 1 - Understanding matrix memory layout and matrix multiplication

In this first part, we will analyze the memory layout datas and the matrix multiplication algorithm.

### 3.1. Memory layout

The memory layout is an important aspect of the matrix multiplication algorithm. 
The best optimization is to have two matrixes with one in `row-major` order and the other in `column-major` order.

With this layout, the algorithm can access the memory in a linear way, which is the best way to use the cache memory.

In our case we have two matrixes `A` and `B` in column-major order and the result matrix `C` in row-major order.

Column-major representation with indexes for a `3 x 4` matrix in memory:

$$ {Memory} = \begin{bmatrix} 0 & 3 & 6 & 9 & 1 & 4 & 7 & 10 & 2 & 5 & 8 & 11 \end{bmatrix} $$

Row-major representation with indexes for a `3 x 4` matrix in memory:

$$ {Memory} = \begin{bmatrix} 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & 11 \end{bmatrix} $$

Representation of a column major table 2D `3 x 4`:

$$ {Matrix} = \begin{bmatrix} 0 & 3 & 6 & 9 \\ 1 & 4 & 7 & 10 \\ 2 & 5 & 8 & 11 \end{bmatrix} $$

Representation of a row major table 2D `3 x 4`:

$$ {Matrix} = \begin{bmatrix} 0 & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8 \\ 9 & 10 & 11 \end{bmatrix} $$

From this analysis, we can see that the switch from column-major to row-major order is a simple transposition of the matrix.

$$ {ColumnMajor} = {RowMajor}^T $$

### 3.2. Matrix multiplication algorithm

For a simple matrix $N \times N$ multiplication algorithm with range of data in row-major memory. 

```c
void naive_square_matrix_multiplication(const int n, const double *matrix_a,
					const double *matrix_b,
					double *matrix_c)
{
	// row i of matrix_a
	for (int i = 0; i < n; ++i) {
		// column j of matrix_b
		for (int j = 0; j < n; ++j) {
			// calculate cij
			double cij = 0.0;

			for (int k = 0; k < n; ++k) {
				cij += matrix_a[i + k * n] *
				       matrix_b[k + j * n];
			}
			matrix_c[i + j * n] = cij;
		}
	}
}

``` 

## 4. Stage 2 - Source template and compiler flags

For this session, we will compile the code without any optimization flags.

```sh
gcc -o main main.c
```
In the next part of the lab, we will implement the tiling technique to optimize the matrix multiplication algorithm.

We have these commands to run the code:

```sh
# 100x100 matrix multiplication
./main 100
# 100x100 tiling matrix multiplication with tile size 5x5
./main 100 5
```
<legends> source: CNM_lab01.etape1.pdf </legends>

The output of the code will be the time taken to execute the matrix multiplication algorithm in `seconds`.

## 5. Stage 3 - Implementing general matrix multiplication

In this stage, we will implement the general matrix multiplication algorithm to multiply two matrixes $A = M \times N$ and $B = N \times K$ to obtain the matrix $C = M \times K$.

$$ A \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1N} \\ a_{21} & a_{22} & \cdots & a_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ a_{M1} & a_{M2} & \cdots & a_{MN} \end{bmatrix} \times B \begin{bmatrix} b_{11} & b_{12} & \cdots & b_{1K} \\ b_{21} & b_{22} & \cdots & b_{2K} \\ \vdots & \vdots & \ddots & \vdots \\ b_{N1} & b_{N2} & \cdots & b_{NK} \end{bmatrix} = C \begin{bmatrix} c_{11} & c_{12} & \cdots & c_{1K} \\ c_{21} & c_{22} & \cdots & c_{2K} \\ \vdots & \vdots & \ddots & \vdots \\ c_{M1} & c_{M2} & \cdots & c_{MK} \end{bmatrix} $$

To realize this, we will implement the following function:

```c
ANDRE CODE HERE
```

With this function, we realize that in one of the matrixes we need to jump `N` elements to go to the next row. This jump will impact the cache memory behavior.

## 6. Stage 4 - Measuring naïve matrix multiplication performance

Now that we have implemented the general matrix multiplication algorithm, we will measure the performance of the algorithm.

We can mesure the time taken to execute the matrix multiplication algorithm in `seconds`.

To measure the performance of the algorithm, we will use `perf.py` to run the code with different matrix sizes and tile sizes.

```sh
python3 perf.py --start 10 -e 1000 -i 100 -T -S -F naive10-1000.svg
```
Here is the output of the script:

@import "../perf_plots/naive10-1000.svg"



## 9. Conclusion

## 10. Ref

- ChatGPT for the help to make the `perf.py` script


