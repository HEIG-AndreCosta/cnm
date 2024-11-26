# Lab04 - Session 01 - Introduction to CUDA
Follow the steps of the laboratory and complete the information

## Stage 3

Fill the table with the information form the cabilities application

| Field                     | Value |
|---------------------------|-------|
| Device Name               |       |
| CUDA driver version       |       |
| CUDA runtime version      |       |
| CUDA Capability version   |       |
| Multiprocesors (MP)       |       |
| CUDA cores/MP             |       |
| Total CUDA cores          |       |
| GPU Max clock rate        |       |
| Global Memory             |       |
| Shared memory/block       |       |
| Registers/block           |       |
| L2 Cache size             |       |
| Warp size                 |       |
| Max threads/block         |       |
| Max dim thread block      |       |

Answer these questions using the [CUDA C++ programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

* What is a Streaming multiprocesor?
* What is a wrap?
* What is a block?
* What is a grid?

## Stage 4

Answer these questions searching in the source code.

* What specifier is used in the function "vectorAdd" to declare it as CUDA kernel?
* How many threads per block is it using?
* How many blocks are used in total?
* How many threads are used in total?
* What function is used to allocate memory in the device? and to free memory in the device?
* What function is used to copy memory from the host to the device? and from the device to the host?

## Stage 4&5 

Provide the timeline graph of your application, and a 2-3 sentences analysis of what's happening during the run. For example, you could answer the following questions:

* From the GPU activities, how much percentage of GPU time is use to perform the actual vector addition?
* From the API calls, What it is the funtion that consume more time?
