#define N 325000000    
#define M 1024 
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void add(float *a, float *b, float *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) c[index] = a[index] + b[index];
}



void random_floats(float* x, int count) {
    for (int i = 0; i < count; i++) 
        x[i] = (rand() % 1000 - 500) / 1000.0;
}



void check_results(float* x, float* y, float* z, int count) {
    for (int i = 0; i < count; i++) 
        if (abs(x[i] + y[i] - z[i]) > pow(10, -8))
            return;

    printf("Check completed correctly\n");
}



int main() {
    float time_transaction1 = 0;
    float time_transaction2 = 0;
    float time_calculation = 0;
    cudaEvent_t start_transaction1, stop_transaction1, start_transaction2, stop_transaction2, start_calculation, stop_calculation;
    cudaEventCreate(&start_transaction1);
    cudaEventCreate(&stop_transaction1);
    cudaEventCreate(&start_transaction2);
    cudaEventCreate(&stop_transaction2);
    cudaEventCreate(&start_calculation);
    cudaEventCreate(&stop_calculation);



    int size = N * sizeof(float);    
    
    float *gpu_a;
    float *gpu_b;
    float *gpu_c;
    
    float *a = new float[size];
    float *b = new float[size];
    float *c = new float[size];



    cudaMalloc((void**)&gpu_a, size);
    cudaMalloc((void**)&gpu_b, size);
    cudaMalloc((void**)&gpu_c, size);



    random_floats(a, N); 
    random_floats(b, N);



    cudaEventRecord(start_transaction1, 0);
    cudaMemcpy(gpu_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_transaction1, 0);



    cudaEventRecord(start_calculation, 0);
    add <<<(N + M - 1) / M, M >>> (gpu_a, gpu_b, gpu_c);
    cudaEventRecord(stop_calculation, 0);



    cudaEventRecord(start_transaction2, 0);
    cudaMemcpy(c, gpu_c, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_transaction2, 0);



    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time_transaction1, start_transaction1, stop_transaction1);
    cudaEventElapsedTime(&time_transaction2, start_transaction2, stop_transaction2);
    cudaEventElapsedTime(&time_calculation, start_calculation, stop_calculation);



    check_results(a, b, c, N);



    printf("Elapsed time_transaction1: %.2f ms\n", time_transaction1);
    printf("Elapsed time_transaction2: %.2f ms\n", time_transaction2);
    printf("Elapsed time_calculation:  %.2f ms\n", time_calculation);



    cudaEventDestroy(start_transaction1);
    cudaEventDestroy(stop_transaction1);
    cudaEventDestroy(start_transaction2);
    cudaEventDestroy(stop_transaction2);
    cudaEventDestroy(start_calculation);
    cudaEventDestroy(stop_calculation);

    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    delete[] a;
    delete[] b;
    delete[] c;



    return 0;
}

