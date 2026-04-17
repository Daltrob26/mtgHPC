#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include "utils.h"
#include "kmeanKernel.cuh"


std::vector<int> kMeansCUDA(
    const std::vector<Card>& data,                        
    int k, 
    int max_iters,
    const int blocks=126
) {

    int n = data.size();
    int dim = data[0].features.size();

    std::vector<double> flat(n * dim);

    for (int i = 0; i < n; i++)
        std::memcpy(flat.data() + i * dim,
                    data[i].features.data(),
                    dim * sizeof(double));

    std::vector<int> labels(n);

    std::vector<double> centroids(k * dim);

    // seed the random starting cards to be the centroids
    //  08051993 because thats mtg's birthday
    std::mt19937 rng(851993);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int c = 0; c < k; c++) {
        int idx = dist(rng);
        // print the names of cards used for centroids
        //  std::cout << data[idx].name << "\n";
        std::memcpy(&centroids[c * dim],
                    data[idx].features.data(),
                    dim * sizeof(double));
    }

    // GPU memory
    double *d_data, *d_centroids;
    int *d_labels;

    double *d_contribSums;
    int *d_contribCounts;

    double *d_sums;
    int *d_counts;

    cudaMalloc(&d_data, n * dim * sizeof(double));
    cudaMalloc(&d_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_labels, n * sizeof(int));

    cudaMalloc(&d_contribSums, n * dim * sizeof(double));
    cudaMalloc(&d_contribCounts, n * sizeof(int));

    cudaMalloc(&d_sums, k * dim * sizeof(double));
    cudaMalloc(&d_counts, k * sizeof(int));

    cudaMemcpy(d_data, flat.data(),
               n * dim * sizeof(double),
               cudaMemcpyHostToDevice);

    dim3 block(blocks);
    dim3 grid((n + block.x - 1) / block.x);

    // main loop
    for (int it = 0; it < max_iters; it++) {
        cudaMemcpy(d_centroids, 
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
            n, dim);
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
        std::vector<double> hostSums(k * dim);
        std::vector<int> hostCounts(k);

        cudaMemcpy(hostSums.data(), d_sums,
                   k * dim * sizeof(double),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(hostCounts.data(), d_counts,
                   k * sizeof(int),
                   cudaMemcpyDeviceToHost);

        for (int c = 0; c < k; c++) {
            if (hostCounts[c] == 0) 
                continue;
            for (int d = 0; d < dim; d++) {
                centroids[c * dim + d] = hostSums[c * dim + d] / hostCounts[c];
            }
        }
    }

    cudaMemcpy(labels.data(), d_labels,
               n * sizeof(int),
               cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_contribSums);
    cudaFree(d_contribCounts);
    cudaFree(d_sums);
    cudaFree(d_counts);

    return labels;
}


int main() {
    std::ifstream infile("mtg_features.csv");
    std::string headerLine;
    std::getline(infile, headerLine);

    if (!headerLine.empty() &&
        (headerLine.back() == '\n' || headerLine.back() == '\r')) {
        headerLine.pop_back();
    }

    std::vector<std::string> header = parseCSVRow(headerLine);

    int k = 5;
    int iter = 100;

    
    std::vector<int> labels;
    auto data = readCSV("mtg_features.csv");
    
    for (int blocks = 64; blocks <= 1024; blocks *= 2) {
        double totalTime = 0.0;
        for (int run = 0; run < NUM_RUNS; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
    
            labels = kMeansCUDA(data, k, iter, blocks);
    
            // sync before stopping the clock
            cudaDeviceSynchronize();
    
            auto end = std::chrono::high_resolution_clock::now();
    
            std::chrono::duration<double> elapsed = end - start;
            totalTime += elapsed.count();
    
            // std::cout << "Run " << run + 1 << " completed in "
            //           << elapsed.count() << " seconds.\n";
        }
    
        double averageTime = totalTime / NUM_RUNS;
    
        std::cout << "Average time over " << NUM_RUNS
                  << " runs with block size "<< blocks 
                  << ": " << averageTime << " seconds.\n";
    }

    writeCSVWithCardData("clusteredCardsCuda.csv", data, labels, header);

    return 0;
}
