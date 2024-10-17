<style>
  legends {
    font-size: 0.8em;
    color: #333;
  }
</style>

<img src="heig-vd.png" style="height:80px;">

# <center> Labo 01 {ignore=true}

# <center> Session 02

# <center> Matrix multiplication optimization {ignore=true}

## <center>Departement : TIC {ignore=true}

## <center>CNM{ignore=true}

Author: **Andre Costa & Alexandre Iorio**

Professor: **Marina Zapater**

Assistant : **Mehdi Akeddar**

Classroom : **A09**

Date : **16.10.2024**

<!-- pagebreak -->

<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=4 orderedList=false} -->

## Table des matières {ignore=true}

<!-- code_chunk_output -->

- [0. Preface](#0-preface)
- [1. Introduction](#1-introduction)
- [2. Material](#2-material)
- [3. Stage 3](#3-stage-3)
  - [Compilation](#compilation)
  - [Results](#results)
- [9. Conclusion](#9-conclusion)

<!-- /code_chunk_output -->

<!-- pagebreak -->

## 0. Preface

This lab is written in Markdown format optimized for the interpreter used by the `Markdown Preview Enhanced` plugin of `Visual Studio Code`.

## 1. Introduction

This lab focuses on optimizing matrix multiplication and edge detection using convolution, key operations in image processing. The goal is to improve performance by considering cache memory hierarchy and applying loop unrolling techniques. We will test different compiler optimizations and measure the impact on execution time.

## 2. Material

To realize this laboratory, we will use the an `Nvidia® Jetson Orin Nano`.

To realize measurements, we will use the return of the `main` function to measure the time to process an edge detection with differents kernels on an image.

## 3. Stage 3

In this first part, we will measure the time with a naïve algorithme.


Measure convolution time with the images inside /images

| Image                   | Rows | Columns | time [us] |
| :---------------------- | :--: | :-----: | --------: |
| images/bike.jpg         | 480  |   640   |    34'598 |
| images/bike_edges.png   | 480  |   640   |    34'700 |
| images/coins.png        | 246  |   300   |     8'258 |
| images/coins_edges.png  | 246  |   300   |     8'254 |
| images/engine.png       | 480  |   640   |    34'658 |
| images/coins_y.png      | 246  |   300   |     8'296 |
| images/engine_y.png     | 480  |   640   |    34'661 |
| images/bike_x.png       | 480  |   640   |    34'649 |
| images/engine_edges.png | 480  |   640   |    34'678 |
| images/bike_y.png       | 480  |   640   |    34'760 |
| images/engine_x.png     | 480  |   640   |    34'784 |
| images/coins_x.png      | 246  |   300   |     8'301 |

With this methods, we have an average process time per pixel of ~ 0.11 [us]. 

# 4 stage 4 - unrolling the loop 

In this stage, we will unroll the loop that iterates over the kernel 3x3 with simple `#define`

```c++
for (int i = 1; i < image_rows - 1; ++i) {
		for (int j = 1; j < image_cols - 1; ++j) {
			Gx[i][j] += sobel_x[0][0] * A[i - 1][j - 1];
			Gy[i][j] += sobel_y[0][0] * A[i - 1][j - 1];
			Gx[i][j] += sobel_x[0][1] * A[i - 1][j];
			Gy[i][j] += sobel_y[0][1] * A[i - 1][j];
			Gx[i][j] += sobel_x[0][2] * A[i - 1][j + 1];
			Gy[i][j] += sobel_y[0][2] * A[i - 1][j + 1];

			Gx[i][j] += sobel_x[1][0] * A[i][j - 1];
			Gy[i][j] += sobel_y[1][0] * A[i][j - 1];
			Gx[i][j] += sobel_x[1][1] * A[i][j];
			Gy[i][j] += sobel_y[1][1] * A[i][j];
			Gx[i][j] += sobel_x[1][2] * A[i][j + 1];
			Gy[i][j] += sobel_y[1][2] * A[i][j + 1];

			Gx[i][j] += sobel_x[2][0] * A[i + 1][j - 1];
			Gy[i][j] += sobel_y[2][0] * A[i + 1][j - 1];
			Gx[i][j] += sobel_x[2][1] * A[i + 1][j];
			Gy[i][j] += sobel_y[2][1] * A[i + 1][j];
			Gx[i][j] += sobel_x[2][2] * A[i + 1][j + 1];
			Gy[i][j] += sobel_y[2][2] * A[i + 1][j + 1];
		}
	}
```

At this point we can see that the time to process an image with an unrolling loop.

| Image                   | Rows | Columns | time [us] (delta) | Difference with naïve algo |
| ----------------------- | :--: | :-----: | ----------------: | -------------------------: |
| images/bike.jpg         | 480  |   640   |            24'077 |                     10'521 |
| images/bike_edges.png   | 480  |   640   |            24'159 |                     10'541 |
| images/coins.png        | 246  |   300   |             5'754 |                      2'504 |
| images/coins_edges.png  | 246  |   300   |             5'783 |                      2'471 |
| images/engine.png       | 480  |   640   |            24'229 |                     10'429 |
| images/coins_y.png      | 246  |   300   |             5'770 |                      2'526 |
| images/engine_y.png     | 480  |   640   |            24'149 |                     10'512 |
| images/bike_x.png       | 480  |   640   |            24'322 |                     10'327 |
| images/engine_edges.png | 480  |   640   |            24'180 |                     10'498 |
| images/bike_y.png       | 480  |   640   |            24'221 |                     10'539 |
| images/engine_x.png     | 480  |   640   |            24'250 |                     10'534 |
| images/coins_x.png      | 246  |   300   |             5'797 |                      2'504 |

With this methods, we have an average process time per pixel of ~ 0.078 [us]. We can see that the unrolling loop has a significant impact on the processing time about 30% faster than the naïve algorithm.

# 5 stage 5 - compiler optimization

In addition to what was required for this step, we also compared the performance of our loop unrolling implementation with -O0, -O1, and -O2.

In total, we compared 7 different configurations:

- No loop unrolling
- Manual loop unrolling with -O0
- Manual loop unrolling with -O1
- Manual loop unrolling with -O2
- Loop unrolling (-funroll-loops) with -O0
- Loop unrolling (-funroll-loops) with -O1
- Loop unrolling (-funroll-loops) with -O2

### Compilation

Once the Makefile is modified, running the make command allows compiling the different configurations:

```bash
make -j16
g++ -O0 -c -Wall -I /usr/include/opencv4 -o edge_detection_no_unroll.o edge_detection.cpp
g++ -O0 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING -o edge_detection_manualo0_unroll.o edge_detection.cpp
g++ -O1 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING -o edge_detection_manualo1_unroll.o edge_detection.cpp
g++ -O2 -c -Wall -I /usr/include/opencv4 -DLOOP_UNROLLING -o edge_detection_manualo2_unroll.o edge_detection.cpp
g++ -O0 -c -Wall -I /usr/include/opencv4 -funroll-loops -o edge_detection_compilero0_unroll.o edge_detection.cpp
g++ -O1 -c -Wall -I /usr/include/opencv4 -funroll-loops -o edge_detection_compilero1_unroll.o edge_detection.cpp
g++ -O2 -c -Wall -I /usr/include/opencv4 -funroll-loops -o edge_detection_compilero2_unroll.o edge_detection.cpp
g++ edge_detection_no_unroll.o -o edge_detection_no_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_manualo0_unroll.o -o edge_detection_manualo0_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_compilero0_unroll.o -o edge_detection_compilero0_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_manualo1_unroll.o -o edge_detection_manualo1_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_compilero2_unroll.o -o edge_detection_compilero2_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_compilero1_unroll.o -o edge_detection_compilero1_unroll -lopencv_imgcodecs -lopencv_core
g++ edge_detection_manualo2_unroll.o -o edge_detection_manualo2_unroll -lopencv_imgcodecs -lopencv_core
```

### Results

| Image                   | Rows | Columns | No Loop Unrolling | Conv. time (SW loop unrolling) (-O0) | Conv. time (SW loop unrolling) (-O1) | Conv. time (SW loop unrolling) (-O2) | Conv. time (compiler -O0) | Conv. time (compiler -O1) | Conv. time (compiler -O2) |
| ----------------------- | ---- | ------- | ----------------- | ------------------------------------ | ------------------------------------ | ------------------------------------ | ------------------------- | ------------------------- | ------------------------- |
| images/bike.jpg         | 480  | 640     | 34598 (+0)        | 24077 (-10521)                       | 4992 (-29606)                        | 4897 (-29701)                        | 34537 (-61)               | 5013 (-29585)             | 4924 (-29674)             |
| images/bike_edges.png   | 480  | 640     | 34700 (+0)        | 24159 (-10541)                       | 5006 (-29694)                        | 4935 (-29765)                        | 35104 (+404)              | 5033 (-29667)             | 4933 (-29767)             |
| images/coins.png        | 246  | 300     | 8258 (+0)         | 5754 (-2504)                         | 1182 (-7076)                         | 1190 (-7068)                         | 8244 (-14)                | 1219 (-7039)              | 1173 (-7085)              |
| images/coins_edges.png  | 246  | 300     | 8254 (+0)         | 5783 (-2471)                         | 1190 (-7064)                         | 1169 (-7085)                         | 8269 (+15)                | 1211 (-7043)              | 1178 (-7076)              |
| images/engine.png       | 480  | 640     | 34658 (+0)        | 24229 (-10429)                       | 5011 (-29647)                        | 4950 (-29708)                        | 35036 (+378)              | 5062 (-29596)             | 4979 (-29679)             |
| images/coins_y.png      | 246  | 300     | 8296 (+0)         | 5770 (-2526)                         | 1192 (-7104)                         | 1177 (-7119)                         | 8296 (+0)                 | 1202 (-7094)              | 1185 (-7111)              |
| images/engine_y.png     | 480  | 640     | 34661 (+0)        | 24149 (-10512)                       | 5009 (-29652)                        | 4937 (-29724)                        | 34604 (-57)               | 5039 (-29622)             | 4962 (-29699)             |
| images/bike_x.png       | 480  | 640     | 34649 (+0)        | 24322 (-10327)                       | 5105 (-29544)                        | 4974 (-29675)                        | 34759 (+110)              | 5105 (-29544)             | 4939 (-29710)             |
| images/engine_edges.png | 480  | 640     | 34678 (+0)        | 24180 (-10498)                       | 5101 (-29577)                        | 4964 (-29714)                        | 35435 (+757)              | 5121 (-29557)             | 4989 (-29689)             |
| images/bike_y.png       | 480  | 640     | 34760 (+0)        | 24221 (-10539)                       | 5093 (-29667)                        | 5021 (-29739)                        | 34758 (-2)                | 5071 (-29689)             | 5007 (-29753)             |
| images/engine_x.png     | 480  | 640     | 34784 (+0)        | 24250 (-10534)                       | 5134 (-29650)                        | 4994 (-29790)                        | 34704 (-80)               | 5099 (-29685)             | 4992 (-29792)             |
| images/coins_x.png      | 246  | 300     | 8301 (+0)         | 5797 (-2504)                         | 1217 (-7084)                         | 1192 (-7109)                         | 8288 (-13)                | 1218 (-7083)              | 1191 (-7110)              |



## 9. Conclusion

With the `-funroll-loops` option at `-O0`, we observe that the performance remains the same as the configuration without loop unrolling. This indicates that the compiler does not apply loop unrolling at the `-O0` optimization level.

At `-O1` and `-O2`, we see that the performance of manually unrolled loops matches that of the compiler's loop unrolling. This suggests that the loop optimization performed by the compiler is equivalent to our manual implementation.

However, at `-O0`, we achieve significant performance gains by manually applying loop unrolling, as the compiler does not perform this optimization at that level.

Overall, loop unrolling proves to enhance the performance of our convolution implementation.

