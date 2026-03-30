#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>


// struct to hold card data
//  namely to seperate the name from the numerical data
struct Card {
  std::string name;
  std::vector<double> features;
};

// parses one row of the datasheet,
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

//write out the data, final col is the cluster the card belongs to
void writeCSVWithCardData(const std::string &filename, const std::vector<Card> &data, const std::vector<int> &labels, const std::vector<std::string> &header) 
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing.\n";
        return;
    }

    //header
    for (size_t i = 0; i < header.size(); ++i) {
        out << '"' << header[i] << '"';
        out << ",";
    }
    out << "\"cluster\"\n";

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

// compute the distance between two vector points (card and centroid)
double computeDistance(const std::vector<double> &a, const std::vector<double> &b) {
  double sum = 0.0;
  for (int i = 0; i < a.size(); ++i) {
    double difference = a[i] - b[i];
    sum += difference * difference;
  }
  return std::sqrt(sum);
}

std::vector<int> kMeans(const std::vector<Card> &data, int k, int max_interations) {
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
    //print the names of cards used for centroids
    // std::cout << data[index].name << "\n";
    centroids.push_back(data[index].features);
  }


  //main loop (parallelize this one)
  for (int iterations = 0; iterations < max_interations; iterations++){
    //assign cards to centroids
    for (int cardNum = 0; cardNum < n; ++cardNum){
        double shortestDistance = 1e18;
        int bestCluster = 0;
        for (int cluster = 0; cluster < k; ++cluster){
            double distance = computeDistance(data[cardNum].features, centroids[cluster]);
            if (distance < shortestDistance){
                shortestDistance = distance;
                bestCluster = cluster;
            }
        }    
        labels[cardNum] = bestCluster;
    }

    //recompute centroids

    std::vector<std::vector<double>> newCentroids(k, std::vector<double>(dimensions, 0.0));
    std::vector<int> counts(k, 0);

    //take the average of the features of each card in a group

    for (int i =0; i < n; ++i){
        int cluster = labels[i];
        counts[cluster]++;

        for (int d = 0; d < dimensions; ++d){
            newCentroids[cluster][d] += data[i].features[d];
        }
    }

    for (int c = 0; c < k; ++c){
        if (counts[c] == 0) continue;
        for (int d = 0; d < dimensions; ++d){
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
    if (!headerLine.empty() && (headerLine.back() == '\n' || headerLine.back() == '\r')) {
        headerLine.pop_back();
    }
    std::vector<std::string> header = parseCSVRow(headerLine);

    auto data = readCSV("mtg_features.csv");

    int k = 5;
    int iter = 100;
    auto start = std::chrono::high_resolution_clock::now();
    auto labels = kMeans(data, k, iter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "K-means completed in " << elapsed.count() << " seconds.\n";

    writeCSVWithCardData("clusteredCards.csv", data, labels, header);

    return 0;
}