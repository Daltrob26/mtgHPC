#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

struct Card {
    std::string name;
    std::vector<double> features;
};

std::vector<std::string> parseCSVRow(const std::string &line) {
    std::vector<std::string> result;
    std::string cur;
    bool inQuotes = false;

    for (char c : line) {
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            result.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    result.push_back(cur);
    return result;
}

std::vector<Card> readCSV(const std::string &filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Card> data;

    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }

        auto cols = parseCSVRow(line);

        Card card;
        card.name = cols[0];

        for (size_t i = 1; i < cols.size(); ++i) {
            card.features.push_back(std::stod(cols[i]));
        }

        data.push_back(card);
    }

    return data;
}

//write out the data, final col is the cluster the card belongs to
void writeCSVWithCardData(const std::string &filename, const std::vector<Card> &data, const std::vector<int> &labels) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    // Write rows
    for (size_t i = 0; i < data.size(); ++i) {
        out << '"' << data[i].name << '"';
        for (double f : data[i].features) {
            out << "," << f;
        }
        out << "," << labels[i] << "\n";
    }

    out.close();
}

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

    double bestDist = 1e18;
    int bestCluster = 0;

    for (int c = 0; c < k; ++c) {
        double sum = 0.0;

        for (int d = 0; d < dim; ++d) {
            double diff = data[i * dim + d] - centroids[c * dim + d];
            sum += diff * diff;
        }

        if (sum < bestDist) {
            bestDist = sum;
            bestCluster = c;
        }
    }

    labels[i] = bestCluster;
}

std::vector<int> kMeansCUDA(const std::vector<Card>& data, int k, int max_iters) {
    int n = data.size();
    int dim = data[0].features.size();

    // ---- Flatten data ----
    std::vector<double> data_flat(n * dim);
    for (int i = 0; i < n; ++i) {
        std::memcpy(
            data_flat.data() + i * dim,
            data[i].features.data(),
            dim * sizeof(double)
        );
    }

    std::vector<int> labels(n, 0);

    // ---- Initialize centroids ----
    std::vector<double> centroids(k * dim);

    std::mt19937 rng(851993);
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (int i = 0; i < k; ++i) {
        int idx = dist(rng);
        std::memcpy(
            centroids.data() + i * dim,
            data[idx].features.data(),
            dim * sizeof(double)
        );
    }

    // ---- Device memory ----
    double *d_data = nullptr;
    double *d_centroids = nullptr;
    int *d_labels = nullptr;

    cudaMalloc(&d_data, n * dim * sizeof(double));
    cudaMalloc(&d_centroids, k * dim * sizeof(double));
    cudaMalloc(&d_labels, n * sizeof(int));

    cudaMemcpy(d_data, data_flat.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // ---- Main loop ----
    for (int iter = 0; iter < max_iters; ++iter) {

        cudaMemcpy(d_centroids, centroids.data(), k * dim * sizeof(double), cudaMemcpyHostToDevice);

        assignClusters<<<gridSize, blockSize>>>(
            d_data,
            d_centroids,
            d_labels,
            n,
            k,
            dim
        );

        cudaMemcpy(labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);

        // ---- Recompute centroids (CPU) ----
        std::vector<double> newCentroids(k * dim, 0.0);
        std::vector<int> counts(k, 0);

        for (int i = 0; i < n; ++i) {
            int c = labels[i];
            counts[c]++;

            for (int d = 0; d < dim; ++d) {
                newCentroids[c * dim + d] += data_flat[i * dim + d];
            }
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;

            for (int d = 0; d < dim; ++d) {
                newCentroids[c * dim + d] /= counts[c];
            }
        }

        centroids = newCentroids;
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);

    return labels;
}

int main() {
    auto data = readCSV("mtg_features.csv");

    int k = 5;
    int iter = 100;

    auto start = std::chrono::high_resolution_clock::now();

    auto labels = kMeansCUDA(data, k, iter);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "K-means CUDA completed in "
              << elapsed.count()
              << " seconds.\n";

    writeCSVWithCardData("clusteredCards.csv", data, labels);

    return 0;
}