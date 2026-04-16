#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "utils.h"



// parses one row of the datasheet

double computeDistance(const std::vector<double> &a,
                       const std::vector<double> &b) {
  double sum = 0.0;
  for (int i = 0; i < a.size(); ++i) {
    double diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

std::vector<int> kMeans(const std::vector<Card> &data, int k,
                        int max_iterations) {
  int n = data.size();
  int dimensions = data[0].features.size();

  std::vector<int> labels(n, 0);

  // initialize centroids
  std::vector<std::vector<double>> centroids;
  std::mt19937 rng(851993);
  std::uniform_int_distribution<int> dist(0, n - 1);

  for (int i = 0; i < k; ++i) {
    centroids.push_back(data[dist(rng)].features);
  }

  for (int iter = 0; iter < max_iterations; iter++) {

    // divide cards across threads
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      double bestDist = 1e18;
      int bestCluster = 0;

      for (int c = 0; c < k; ++c) {
        double d = computeDistance(data[i].features, centroids[c]);
        if (d < bestDist) {
          bestDist = d;
          bestCluster = c;
        }
      }
      labels[i] = bestCluster;
    }

    // compute new centroids
    std::vector<std::vector<double>> newCentroids(
        k, std::vector<double>(dimensions, 0.0));
    std::vector<int> counts(k, 0);

#pragma omp parallel
    {
      std::vector<std::vector<double>> localCentroids(
          k, std::vector<double>(dimensions, 0.0));
      std::vector<int> localCounts(k, 0);

#pragma omp for
      for (int i = 0; i < n; ++i) {
        int cluster = labels[i];
        localCounts[cluster]++;
        for (int d = 0; d < dimensions; ++d) {
          localCentroids[cluster][d] += data[i].features[d];
        }
      }

      // slows things down a bit, but ensures we dont step over each other, and since we only have a few K's not really important to distribute
#pragma omp critical
      {
        for (int c = 0; c < k; ++c) {
          counts[c] += localCounts[c];
          for (int d = 0; d < dimensions; ++d) {
            newCentroids[c][d] += localCentroids[c][d];
          }
        }
      }
    }

    // serial, no real point to dividng this part up
    for (int c = 0; c < k; ++c) {
      if (counts[c] == 0)
        continue;
      for (int d = 0; d < dimensions; ++d) {
        newCentroids[c][d] /= counts[c];
      }
    }

    centroids = newCentroids;
  }

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
  auto data = readCSV("mtg_features.csv");

  int k = 5;
  int iter = 100;

  double totalTime = 0.0;

  for (int run = 0; run < NUM_RUNS; ++run) {
    auto start = std::chrono::high_resolution_clock::now();
    auto labels = kMeans(data, k, iter);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    totalTime += elapsed.count();
    std::cout << "Run " << run + 1 << " completed in " << elapsed.count()
              << " seconds.\n";
  }

  double averageTime = totalTime / NUM_RUNS;
  std::cout << "Average time over " << NUM_RUNS << " runs: " << averageTime
            << " seconds.\n";

  auto labels = kMeans(data, k, iter);
  writeCSVWithCardData("clusteredCardsOpenMP.csv", data, labels, header);

  return 0;
}