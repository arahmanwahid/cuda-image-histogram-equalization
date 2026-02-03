
#include <iostream>
#include <cuda_runtime.h>

__global__ void computeHistogram(const unsigned char* data, int* histogram, int size) {
    __shared__ int local_histo[256];
    int tx = threadIdx.x;
    if (tx < 256) local_histo[tx] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(&local_histo[data[tid]], 1);
    }
    __syncthreads();

    if (tx < 256) {
        atomicAdd(&histogram[tx], local_histo[tx]);
    }
}

int main() {
    const int N = 1024 * 1024; 
    unsigned char* h_img = (unsigned char*)malloc(N);
    int* h_hist = (int*)malloc(256 * sizeof(int));

    for (int i = 0; i < N; i++) h_img[i] = (unsigned char)(i % 256);

    unsigned char *d_img; int *d_hist;
    cudaMalloc(&d_img, N);
    cudaMalloc(&d_hist, 256 * sizeof(int));

    cudaMemcpy(d_img, h_img, N, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    computeHistogram<<<(N + 255) / 256, 256>>>(d_img, d_hist, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    long total = 0;
    for (int i = 0; i < 256; i++) total += h_hist[i];

    std::cout << "Total Pixels: " << total << (total == N ? " (SUCCESS)" : " (FAIL)") << std::endl;

    cudaFree(d_img); cudaFree(d_hist);
    free(h_img); free(h_hist);
    return 0;
}
