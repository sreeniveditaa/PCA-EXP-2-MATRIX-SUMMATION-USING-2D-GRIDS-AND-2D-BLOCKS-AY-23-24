# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>NAME : SREE NIVEDITAA SARAVANAN</h3>
<h3>REGISTER NO : 212223230213</h3>
<h3>EX. NO 2</h3>
<h3>DATE : 11.9.2025</h3>
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```c
%%writefile matrix_add.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void addMatrix(int *a, int *b, int *c, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;
    
    if (ix < nx && iy < ny)
        c[idx] = a[idx] + b[idx];
}

int main() {
    int nx = 1 << 10;  // 1024x1024 matrix
    int ny = 1 << 10;
    int nxy = nx * ny;
    size_t size = nxy * sizeof(int);
    
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);
    
    for (int i = 0; i < nxy; i++) {
        h_a[i] = rand() & 0xFF;
        h_b[i] = rand() & 0xFF;
    }
    
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    dim3 block(32, 32);
    dim3 grid((nx + 31) / 32, (ny + 31) / 32);
    
    addMatrix<<<grid, block>>>(d_a, d_b, d_c, nx, ny);
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    printf("h_c[0] = %d (a=%d, b=%d)\n", h_c[0], h_a[0], h_b[0]);
    printf("h_c[100] = %d (a=%d, b=%d)\n", h_c[100], h_a[100], h_b[100]);
    printf("Success!\n");
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
```

## OUTPUT:
<img width="701" height="155" alt="image" src="https://github.com/user-attachments/assets/68758a4f-a81a-447c-86cc-8538adc3733f" />

## RESULT:
The host took 0.824321 seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in 0.007480 seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
