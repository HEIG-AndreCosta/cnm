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

To measure the performance of the process, we will use a script to run the code with different matrix sizes. We will memorize the time taken to execute the multiplication and we trace a graph.


## 9. Conclusion

Au terme de ce laboratoire, nous avons conçu et testé un système de traitement des données audio et vidéo en utilisant la carte DE1-SoC et la librairie Xenomai. Nous avons évalué les performances de notre système en mesurant les périodes et les jitters des différentes tâches.

Pour la partie audio, nous avons démontré que la gestion des priorités est cruciale pour assurer le bon fonctionnement du système. La tâche d'acquisition audio a été priorisée pour garantir une acquisition stable, tandis que les tâches de traitement et de logging ont été ajustées en conséquence.

Dans la partie vidéo, nous avons exploré divers modes de traitement, allant de la simple copie à des opérations plus complexes comme l'application de filtres de niveau de gris et de convolution. Les mesures ont montré que les tâches vidéo, notamment celles impliquant des traitements complexes, sont gourmandes en temps CPU et peuvent nécessiter des ajustements de période pour respecter les conditions d'ordonnançabilité.

L'utilisation des conditions d'ordonnançabilité de Liu et Layland nous a permis de vérifier si notre système pouvait fonctionner correctement sous différentes charges. Nous avons ajusté les périodes des tâches vidéo pour garantir leur ordonnançabilité, même si cela a nécessité des compromis sur la fréquence des images (framerate).

Enfin, nous avons validé notre système en utilisant l'outil de mesure de temps d'exécution, garantissant que les tâches critiques respectaient les contraintes temporelles imposées.

Ce laboratoire nous a permis de mieux comprendre les défis liés à l'ordonnancement de tâches en temps réel et l'importance de la gestion des priorités et des périodes pour assurer un système performant et fiable.


## 10. Ref

- ChatGPT pour la réalisation des formules mathématiques en Latex
- ChatGPT pour l'aide à la compréhension des problèmes liés à la gestion des buffers
- [Markdown Preview Enhanced](https://shd101wyy.github.io/markdown-preview-enhanced/#/) pour la réalisation de ce rapport
- Adaptation de la librairie `time_measurement` réalisée par Colin Jaques et Théodros Mulugeta.
- Adaptation du scripts python `jitter.py` du labo03 et labo04


