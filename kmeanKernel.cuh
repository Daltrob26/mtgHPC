#include <cuda_runtime.h>

__global__
void assignClusters(
    const double* data,
    const double* centroids,
    int* labels,
    int n, 
    int k, 
    int dim
);

__global__
void buildPointContributions(
    const double* data,
    const int* labels,
    double* contribSums,
    int* contribCounts,
    int n, 
    int dim
);

__global__
void reduce(
    const double* contribSums,
    const int* contribCounts,
    double* sums,
    int* counts,
    int n, 
    int k, 
    int dim
);
