#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "utils.h"




// compute the distance between two vector points (card and centroid)
double computeDistance(const std::vector<double> &a,
                       const std::vector<double> &b) {
  double sum = 0.0;
  for (int i = 0; i < a.size(); ++i) {
    double difference = a[i] - b[i];
    sum += difference * difference;
  }
  return std::sqrt(sum);
}

std::vector<int> kMeans(const std::vector<Card> &data, int k,
                        int max_interations) {
  int n = data.size();
  int dimensions = data[0].features.size();

  std::vector<int> labels(n, 0);

  // seed the random starting cards to be the centroids
  //  08051993 because thats mtg's birthday
  std::vector<std::vector<double>> centroids;
  std::mt19937 rng(851993);
  std::uniform_int_distribution<int> dist(0, data.size() - 1);
  for (int i = 0; i < k; ++i) {
    int index = dist(rng);
    // print the names of cards used for centroids
    //  std::cout << data[index].name << "\n";
    centroids.push_back(data[index].features);
  }

  // main loop (parallelize this one)
  for (int iterations = 0; iterations < max_interations; iterations++) {
    // assign cards to centroids
    for (int cardNum = 0; cardNum < n; ++cardNum) {
      double shortestDistance = 1e18;
      int bestCluster = 0;
      for (int cluster = 0; cluster < k; ++cluster) {
        double distance =
            computeDistance(data[cardNum].features, centroids[cluster]);
        if (distance < shortestDistance) {
          shortestDistance = distance;
          bestCluster = cluster;
        }
      }
      labels[cardNum] = bestCluster;
    }

    // recompute centroids

    std::vector<std::vector<double>> newCentroids(
        k, std::vector<double>(dimensions, 0.0));
    std::vector<int> counts(k, 0);

    // take the average of the features of each card in a group

    for (int i = 0; i < n; ++i) {
      int cluster = labels[i];
      counts[cluster]++;

      for (int d = 0; d < dimensions; ++d) {
        newCentroids[cluster][d] += data[i].features[d];
      }
    }

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
        std::cout << "Run " << run + 1 << " completed in " << elapsed.count() << " seconds.\n";
    }

    double averageTime = totalTime / NUM_RUNS;
    std::cout << "Average time over " << NUM_RUNS << " runs: " << averageTime << " seconds.\n";

    
    auto labels = kMeans(data, k, iter);
    writeCSVWithCardData("SerialCards.csv", data, labels, header);

    return 0;
}