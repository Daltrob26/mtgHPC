#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include "utils.h"


__global__
void assignClusters(
    const double* data,
    const double* centroids,
    int* labels,
    int n, 
    int k, 
    int dim
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double best = 1e18;
    int bestC = 0;

    for (int c = 0; c < k; c++) {
        double dist = 0.0;

        for (int d = 0; d < dim; d++) {
            double diff = data[i * dim + d] - centroids[c * dim + d];
            dist += diff * diff;
        }

        if (dist < best) {
            best = dist;
            bestC = c;
        }
    }

    labels[i] = bestC;
}

__global__
void buildPointContributions(
    const double* data,
    const int* labels,
    double* contribSums,
    int* contribCounts,
    int n, 
    int dim
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int c = labels[i];

    contribCounts[i] = c;

    for (int d = 0; d < dim; d++) {
        contribSums[i * dim + d] = data[i * dim + d];
    }
}

__global__
void reduce(
    const double* contribSums,
    const int* contribCounts,
    double* sums,
    int* counts,
    int n, 
    int k, 
    int dim
) {

    int c = blockIdx.x;
    int d = threadIdx.x;

    if (c >= k || d >= dim) return;

    double sum = 0.0;
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (contribCounts[i] == c) {
            sum += contribSums[i * dim + d];

            if (d == 0) {
                count++;
            }
        }
    }

    sums[c * dim + d] = sum;

    if (d == 0) {
        counts[c] = count;
    }
}