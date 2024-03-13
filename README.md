# CSC14120 – PARALLEL PROGRAMMING 
# FINAL PROJECT

## Introduction

In this final project, you will be implementing and optimizing the forward-pass of a
convolutional layer using CUDA. Convolutional layers are the primary building blocks
of convolutional neural networks (CNNs), which are used in many machine learning
tasks like image classification, object detection, natural language processing, and
recommendation systems. In general, CNNs work well on tasks where the data/input
features have some level of spatial relationship.
![LenetImage](https://lh5.googleusercontent.com/84RlneM7JSDYDirUr_ceplL4G3-Peyq5dkLJTe2f-3Bj9KuWZjsH2A9Qq5PO5BRLrVfWGPnI3eQu8RkTPgyeUf9ZOWY9JbptVJy9LceAyHRn-O0kbzprx88yb82a5dnCR7EDP7n0)

*Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf*

Your optimized CUDA implementation of the convolutional layer will be used to
perform inference for layers C1 and C3.
We can use [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) (Mini-DNN) framework for implementing the modified version of LeNet-5. 

We will be using the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), where the inputs to the network will be a batch of 10,000 single channel images, each with dimensions of 86 x 86 pixels. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot etc.) where the inputs to the network will be a single channel images with dimensions of 28 x 28 pixels. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot etc.)
The overall learning objectives for this project are:
Demonstrating command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolutional layer forward pass

## Table of Contents

1. [CPU Implementation](#cpu-implementation)
2. [GPU (Parallelized) Implementation](#gpu-parallelized-implementation)
3. [Optimized GPU (Parallelized) Implementation](#optimized-gpu-parallelized-implementation)
4. [Instructions on how to compile and test code](#instructions-on-how-to-compile-and-test-code)

---
## Input data
The network is tested on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) `t10k-images-idx3-ubyte.gz` and `t10k-labels-idx1-ubyte.gz`  which contains 10,000 single channel images each of dimensions 28x28. The output layer consists of 10 nodes, where each node represents the likelihood of the input belonging to one of the 10 classes (T-shirt, dress, sneaker, boot, etc).

## CPU Implementation
File: `src/layer/custom/cpu-forward.cc`
After implementing the neural network using CPU, the achieved classification accuracy is reported as 0.886 when evaluated with a test batch size of 1000 images. This accuracy metric reflects the model's ability to correctly classify the input images into their respective classes based on the predictions made by the network.  <br>

## GPU (Parallelized) Implementation
File: `src/layer/custom/Parallel_v1.cu`
### Tested on Tesla T4
The neural network was tested on a Tesla T4 GPU, and despite the implementation on a different hardware platform, the classification accuracy remained consistent at 0.886. This evaluation was conducted using a test batch size of 1000 images.

## Optimized GPU (Parallelized) Implementation

The optimized implementation of the neural network's convolutional layer resides in the file `src/layer/custom/Parallel_v2.cu` until `Parallel_v4.cu`. These versions reflect progressive optimizations and improvements made to the parallelized implementation.

### Testing on Tesla T4

Even with the optimized versions (from v2 to v4) and continued testing on the Tesla T4 GPU, the classification accuracy remains constant at 0.886 for a batch size of 1000 images.
### Optimization Methods Employed  
- **Parallel_v2:** - Utilized `atomics add` to optimize the implementation. 
- **Parallel_v3:**  - Implemented `atomics add` along with the utilization of `constant memory` to enhance performance. 
-  **Parallel_v4:** - Leveraged `atomics add`, `constant memory`, `tile shared memory`, and `loop unrolling` to further optimize the implementation.

## Instructions on How to Compile and Test Code  
### Compilation Steps
#### CPU

To compile the code:

1. **Check for existing object files:**
   - If the object files are available for each source file (e.g., `cpu.o`, `network_init.o`, etc.), proceed to the next step.
   - If any object file is missing:
     ```bash
     nvcc --compile filename -o filename.o -I ../libgputk/ -I./
     ```
   - This will compile the missing file into an object file.

2. **Compile the final executable:**
   ```bash
   make cpu
   ```


#### GPU

To compile the code:

1. **Check for existing object files:**
   - If the object files are available for each source file (e.g., `cpu.o`, `network_init.o`, etc.), proceed to the next step.
   - If any object file is missing:
     ```bash
     nvcc --compile filename -o filename.o -I ../libgputk/ -I./
     ```
   - This will compile the missing file into an object file.

2. **Compile the final executable:**
   ```bash 
   make parallel_v1  # Compile version 1
   make parallel_v2  # Compile version 2
   ...
     ```
    ### Testing
    To test the CPU version of the program, use the following command: 
    ```bash
     ./cpu  # Run CPU version with default batch size (10000)
    ```
    If you want to specify a custom batch size, use:
    ```bash
     ./cpu <batch_size> # Replace <batch_size> with your desired batch size
    ```
	  To compile the CPU version of the program and run it with a default batch size of 1000, you can use the following command: 
	  ```bash 
	  make run # Compile and run CPU version with default batch size (1000)
	  ```
    To test the GPU version of the program, use the following command: 
    ```bash
     ./parallel_v<version>  
     # Run GPU <version> (1, 2, 3, 4) with default batch size (10000). Example: ./parallel_v1
    ```
    If you want to specify a custom batch size, use:
    ```bash
     ./parallel_v<version> <batch_size> 
     # Replace <batch_size> with your desired batch size
    ```
	  To compile the GPU version of the program and run it with a default batch size of 1000, you can use the following command: 
	  ```bash 
	  make run_v<version> 
	  # Compile and run GPU <version> (1, 2, 3, 4) with default batch size (1000). Example : make run_v1
	 ```
## Conclusion

The LeNet-5 network, a pioneering convolutional neural network architecture, has been successfully implemented and optimized in this project. Through various parallelization techniques and optimizations, the performance of the network's forward pass has been significantly enhanced.

Notably, several optimization methods were employed to maximize the efficiency of the parallelized code:

1. **Parallelization Strategies**: The network's layers were parallelized using CUDA, harnessing the power of GPU parallel processing for faster computations.

2. **Optimized CUDA Implementations**: Different versions (v1, v2, v3, v4) were created, each incorporating specific optimizations such as atomics add, constant memory utilization, tile shared memory, and loop unrolling, to improve the overall efficiency of the parallelized code.

3. **Testing and Validation**: Each version was thoroughly tested and validated on GPUs, specifically the Tesla T4, maintaining a classification accuracy of 0.886 even with a batch size of 1000, ensuring the correctness and reliability of the optimized implementations.

These optimizations have collectively enhanced the performance of the LeNet-5 network, demonstrating the significant impact of parallelization and optimization strategies in accelerating deep neural network computations.
