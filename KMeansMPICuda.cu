#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "kmeanKernel.cuh"
#include "utils.h"

static bool initialized = false;

static double *d_data = nullptr;
static double *d_centroids = nullptr;
static int *d_labels = nullptr;

static double *d_contribSums = nullptr;
static int *d_contribCounts = nullptr;

static double *d_sums = nullptr;
static int *d_counts = nullptr;


void launch_cuda(
    const std::vector<Card>& data,
    const std::vector<double>& centroids,
    int k,
    int dim,
    std::vector<double>& local_sums,
    std::vector<int>& local_counts,
    std::vector<int>& labels
) {
    int n = data.size();

    if (!initialized) {
        cudaMalloc(&d_data, n * dim * sizeof(double));
        cudaMalloc(&d_centroids, k * dim * sizeof(double));
        cudaMalloc(&d_labels, n * sizeof(int));

        cudaMalloc(&d_contribSums, n * dim * sizeof(double));
        cudaMalloc(&d_contribCounts, n * sizeof(int));

        cudaMalloc(&d_sums, k * dim * sizeof(double));
        cudaMalloc(&d_counts, k * sizeof(int));

        initialized = true;
    }


    // flatten data
    std::vector<double> flat(n * dim);
    for (int i = 0; i < n; i++) {
        std::memcpy(
            flat.data() + i * dim,        
            data[i].features.data(),
            dim * sizeof(double)
        );
    }

    labels.resize(n);
    local_sums.assign(k * dim, 0.0);
    local_counts.assign(k, 0);


    cudaMemcpy(d_data, flat.data(),
               n * dim * sizeof(double),
               cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((n + block.x - 1) / block.x);

    // copy inputs
    cudaMemcpy(
        d_centroids, 
        centroids.data(),
        k * dim * sizeof(double),
        cudaMemcpyHostToDevice
    );
  
    cudaDeviceSynchronize(); 

    // assign cards to centroids
    assignClusters<<<grid, block>>>(
        d_data, 
        d_centroids, 
        d_labels,
        n, 
        k,
        dim
    );
    cudaDeviceSynchronize();

    // recompute centroids
    buildPointContributions<<<grid, block>>>(
        d_data, 
        d_labels,
        d_contribSums, 
        d_contribCounts,
        n, 
        dim
    );
    cudaDeviceSynchronize(); 

    cudaMemset(d_sums, 0, k * dim * sizeof(double));
    cudaMemset(d_counts, 0, k * sizeof(int));

    // reduction
    dim3 clusterGrid(k);
    dim3 featureBlock(dim);
    reduce<<<clusterGrid, featureBlock>>>(
        d_contribSums, 
        d_contribCounts,
        d_sums, 
        d_counts,
        n, 
        k, 
        dim
    );
    cudaDeviceSynchronize(); 

    // copy back
    cudaMemcpy(local_sums.data(), d_sums,
               k * dim * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(local_counts.data(), d_counts,
               k * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(labels.data(), d_labels,
               n * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaMemcpy(labels.data(), d_labels,
               n * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize(); 
}

void cleanup_cuda() {
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_contribSums);
    cudaFree(d_contribCounts);
    cudaFree(d_sums);
    cudaFree(d_counts);
}
