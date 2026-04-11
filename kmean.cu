#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>

int NUM_RUNS = 10;

struct Card {
    std::string name;
    std::vector<double> features;
};

std::vector<std::string> parseCSVRow(const std::string &line) {
    std::vector<std::string> result;
    std::string cur;
    bool inQuotes = false;

    for (char c : line) {
        // handle cards with , in the name
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
        // skip the header line
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

// write out the data, final col is the cluster the card belongs to
void writeCSVWithCardData(const std::string &filename,
                          const std::vector<Card> &data,
                          const std::vector<int> &labels,
                          const std::vector<std::string> &header) {
  std::ofstream out(filename);

  for (size_t i = 0; i < header.size(); ++i) {
    out << '"' << header[i] << '"' << ",";
  }
  out << "\"cluster\"\n";

  for (size_t i = 0; i < data.size(); ++i) {
    out << '"' << data[i].name << '"';
    for (double f : data[i].features) {
      out << "," << f;
    }
    out << "," << labels[i] << "\n";
  }
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

            if (d == 0)
                count++;
        }
    }

    sums[c * dim + d] = sum;

    if (d == 0)
        counts[c] = count;
}

std::vector<int> kMeansCUDA(
    const std::vector<Card>& data,                        
    int k, 
    int max_iters
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

    dim3 block(256);
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
    auto data = readCSV("mtg_features.csv");
    std::string headerLine;
    std::getline(infile, headerLine);

  if (!headerLine.empty() &&
      (headerLine.back() == '\n' || headerLine.back() == '\r')) {
    headerLine.pop_back();
  }

    int k = 5;
    int iter = 100;

    double totalTime = 0.0;

    std::vector<int> labels;

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();

        labels = kMeansCUDA(data, k, iter);

        // sync before stopping the clock
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        totalTime += elapsed.count();

        std::cout << "Run " << run + 1 << " completed in "
                  << elapsed.count() << " seconds.\n";
    }

    double averageTime = totalTime / NUM_RUNS;

    std::cout << "Average time over " << NUM_RUNS
              << " runs: " << averageTime << " seconds.\n";

    writeCSVWithCardData("clusteredCardsCuda.csv", data, labels, header);

    return 0;
}