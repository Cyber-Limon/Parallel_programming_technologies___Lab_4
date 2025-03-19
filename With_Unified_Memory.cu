#define N 325000000
#define M 1024
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



__global__ void add(float *a, float *b, float *c) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) c[index] = a[index] + b[index];
}



float random_floats(float* x, int count) {
    for (int i = 0; i < count; i++) 
        x[i] = (rand() % 1000 - 500) / 1000.0;

    return *x;
}



void check_results(float* x, float* y, float* z, int count) {
    for (int i = 0; i < count; i++) {
        if (abs(x[i] + y[i] - z[i]) > pow(10, -8))
            return;
    }

    printf("Check completed correctly\n");
}



int main() {
    float time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *a, *b, *c;
    int size = N * sizeof(float);
    


    cudaMallocManaged((void**)&a, size);
    cudaMallocManaged((void**)&b, size);
    cudaMallocManaged((void**)&c, size);



    random_floats(a, N); 
    random_floats(b, N);
    


    cudaEventRecord(start, 0);
    add <<<(N + M - 1) / M, M >>> (a, b, c);
    cudaEventRecord(stop, 0);



    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    


    check_results(a, b, c, N);



    printf("Elapsed time: %.2f ms\n", time);



    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);



    return 0;
}

