// Bonus: Courtesy of ChatGPT

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <cstring>
#include <chrono>

// =====================================================
// DATA STRUCTURE
// =====================================================

struct Card {
    std::string name;
    std::vector<float> features;
};

// =====================================================
// CSV INPUT
// =====================================================

std::vector<std::string> parseCSVRow(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool inQuotes = false;

    for (char c : line) {
        if (c == '"') inQuotes = !inQuotes;
        else if (c == ',' && !inQuotes) {
            out.push_back(cur);
            cur.clear();
        } else cur += c;
    }
    out.push_back(cur);
    return out;
}

std::vector<Card> readCSV(const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Card> data;

    bool first = true;

    while (std::getline(file, line)) {
        if (first) { first = false; continue; }

        auto cols = parseCSVRow(line);

        Card c;
        c.name = cols[0];

        for (size_t i = 1; i < cols.size(); i++)
            c.features.push_back(std::stof(cols[i]));

        data.push_back(c);
    }

    return data;
}

// =====================================================
// CSV OUTPUT
// =====================================================

void writeCSVWithLabels(
    const std::string& filename,
    const std::vector<Card>& data,
    const std::vector<int>& labels
) {
    std::ofstream out(filename);

    for (size_t i = 0; i < data.size(); i++) {
        out << '"' << data[i].name << '"';

        for (float f : data[i].features)
            out << "," << f;

        out << "," << labels[i] << "\n";
    }
}

// =====================================================
// CUDA K-MEANS CORE
// =====================================================

__global__
void assignAndAccumulate(
    const float* __restrict__ data,
    const float* __restrict__ centroids,
    int* __restrict__ labels,
    float* blockSums,
    int* blockCounts,
    int n,
    int k,
    int dim
) {
    extern __shared__ float smem[];

    float* s_sums = smem;
    int* s_counts = (int*)&s_sums[k * dim];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    for (int j = tid; j < k * dim; j += blockDim.x)
        s_sums[j] = 0.0f;

    for (int j = tid; j < k; j += blockDim.x)
        s_counts[j] = 0;

    __syncthreads();

    if (i < n) {

        const float* point = &data[i * dim];

        int bestC = 0;
        float bestDist = 1e30f;

        #pragma unroll 4
        for (int c = 0; c < k; c++) {

            const float* centroid = &centroids[c * dim];

            float dist = 0.0f;

            #pragma unroll 4
            for (int d = 0; d < dim; d++) {
                float diff = point[d] - centroid[d];
                dist += diff * diff;
            }

            if (dist < bestDist) {
                bestDist = dist;
                bestC = c;
            }
        }

        labels[i] = bestC;

        atomicAdd(&s_counts[bestC], 1);

        for (int d = 0; d < dim; d++) {
            atomicAdd(&s_sums[bestC * dim + d], point[d]);
        }
    }

    __syncthreads();

    for (int j = tid; j < k * dim; j += blockDim.x)
        atomicAdd(&blockSums[j], s_sums[j]);

    for (int j = tid; j < k; j += blockDim.x)
        atomicAdd(&blockCounts[j], s_counts[j]);
}

__global__
void updateCentroids(
    float* centroids,
    const float* blockSums,
    const int* blockCounts,
    int k,
    int dim
) {
    int c = blockIdx.x;
    int d = threadIdx.x;

    if (c < k && d < dim) {
        int count = blockCounts[c];
        if (count > 0)
            centroids[c * dim + d] =
                blockSums[c * dim + d] / count;
    }
}

// =====================================================
// HOST DRIVER
// =====================================================

std::vector<int> kMeansCUDA(
    const std::vector<Card>& data,
    int k,
    int iters
) {
    int n = data.size();
    int dim = data[0].features.size();

    std::vector<float> flat(n * dim);
    std::vector<int> labels(n);

    for (int i = 0; i < n; i++)
        memcpy(&flat[i * dim],
               data[i].features.data(),
               dim * sizeof(float));

    std::vector<float> centroids(k * dim);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (int c = 0; c < k; c++) {
        int idx = dist(rng);
        memcpy(&centroids[c * dim],
               data[idx].features.data(),
               dim * sizeof(float));
    }

    // ---------------- GPU MEMORY ----------------

    float *d_data, *d_centroids, *d_blockSums;
    int *d_labels, *d_blockCounts;

    cudaMalloc(&d_data, n * dim * sizeof(float));
    cudaMalloc(&d_centroids, k * dim * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_blockSums, k * dim * sizeof(float));
    cudaMalloc(&d_blockCounts, k * sizeof(int));

    cudaMemcpy(d_data, flat.data(),
               n * dim * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    size_t sharedSize =
        k * dim * sizeof(float) + k * sizeof(int);

    dim3 centroidGrid(k);
    dim3 centroidBlock(dim);

    // =================================================
    // ITERATIONS
    // =================================================

    for (int it = 0; it < iters; it++) {

        cudaMemcpy(d_centroids,
                   centroids.data(),
                   k * dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaMemset(d_blockSums, 0, k * dim * sizeof(float));
        cudaMemset(d_blockCounts, 0, k * sizeof(int));

        assignAndAccumulate<<<grid, block, sharedSize>>>(
            d_data,
            d_centroids,
            d_labels,
            d_blockSums,
            d_blockCounts,
            n, k, dim
        );

        updateCentroids<<<centroidGrid, centroidBlock>>>(
            d_centroids,
            d_blockSums,
            d_blockCounts,
            k,
            dim
        );

        cudaDeviceSynchronize();

        cudaMemcpy(centroids.data(),
                   d_centroids,
                   k * dim * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(labels.data(),
               d_labels,
               n * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_blockSums);
    cudaFree(d_blockCounts);

    return labels;
}

// =====================================================
// MAIN
// =====================================================

int main() {
    auto data = readCSV("mtg_features.csv");

    int k = 5;
    int iters = 100;

    auto start = std::chrono::high_resolution_clock::now();

    auto labels = kMeansCUDA(data, k, iters);

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "K-means CUDA took "
              << std::chrono::duration<double>(end - start).count()
              << " seconds\n";

    writeCSVWithLabels("clusteredCards.csv", data, labels);

    return 0;
}