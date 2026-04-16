#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <mpi.h>

int NUM_RUNS = 10;

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

// write out the data, final col is the cluster the card belongs to
void writeCSVWithCardData(const std::string &filename,
                          const std::vector<Card> &data,
                          const std::vector<int> &labels,
                          const std::vector<std::string> &header) {
  std::ofstream out(filename);
  if (!out.is_open()) {
    std::cerr << "Failed to open " << filename << " for writing.\n";
    return;
  }

  // header
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
double computeDistance(const std::vector<double> &a,
                       const std::vector<double> &b) {
  double sum = 0.0;
  for (int i = 0; i < a.size(); ++i) {
    double difference = a[i] - b[i];
    sum += difference * difference;
  }
  return std::sqrt(sum);
}

std::vector<int> kMeans(MPI_Datatype* local_card_array, double local_n, int dimensions, int k,
                        int max_interations, int my_rank, MPI_Comm comm) {
  int n = local_n;

  std::vector<int> labels(n, 0);

  // seed the random starting cards to be the centroids
  //  08051993 because thats mtg's birthday
  std::vector<double> my_centroid;
  std::vector<std::vector<double>> centroids;
  double* mpi_centroids = new double[k * dimensions];
  std::mt19937 rng(851993);
  std::uniform_int_distribution<int> dist(0, data.size() - 1);

    if (my_rank < k) {
      int index = dist(rng);
      my_centroid = data[index].features;

      MPI_Send(my_centroid, dimensions, MPI_DOUBLE, 0, my_rank, comm);
    }

     for (int i = 0; i < k; ++i) {
      if (my_rank == 0) {
        MPI_Recv(&mpi_centroids[i * dimensions], dimensions, MPI_DOUBLE, i, i, comm, MPI_STATUS_IGNORE);
      }
    }

    for (int i = 0; i < k; ++i) {
      int index = dist(rng);
      // print the names of cards used for centroids
      //  std::cout << data[index].name << "\n";
      centroids.push_back(data[index].features);
    }

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

void Build_mpi_type(double* features, int card_id, MPI_datatype* mpi_type_card) {
  int array_of_blocklengths[2] = {1, 1}; // name and features
  MPI_Datatype array_of_types[2] = {MPI_INT, MPI_DOUBLE};
  MPI_Aint id_addr, features_addr;
  MPI_Get_address(&card_id, &id_addr);
  MPI_Get_address(features, &features_addr);
  MPI_Aint array_of_displacements[2] = {id_addr, features_addr};

  MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, mpi_type_card);
  MPI_Type_commit(mpi_type_card);


}

int main() {

    // MPI variables
    MPI_Comm comm;
    int my_rank, comm_sz;
    int* counts_array;
    int* dspls_array;

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

    double totalTime = 0.0;
    int dimensions;

    
    MPI_Datatype* mpi_card_array = new MPI_Datatype[n];
    MPI_Datatype* local_card_array;

    MPI_Init();
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    
    


    if (my_rank == 0) {
      // read in the data
      auto data = readCSV("mtg_features.csv");

      // convert the data to a format we can send with MPI
      int n = data.size();
      dimensions = data[0].features.size();
      double* features = new double[n * dimensions];
      for (int i = 0; i < n; i++) {
        for (int d = 0; d < dimensions; ++d) {
          features[i * dimensions + d] = data[i].features[d];
        }
        MPI_Datatype mpi_type_card;
        Build_mpi_type(&features[i * dimensions], i, &mpi_type_card);
        mpi_card_array[i] = mpi_type_card;
      }

      // calculate counts and displacements for scatterv
      counts_array = new int[comm_sz];
      dspls_array = new int[comm_sz];
      for (int i = 0; i < comm_sz; i++) {
        counts_array[i] = n / comm_sz;
        if (i == comm_sz - 1) {
          counts_array[i] += n % comm_sz;
        }
        dspls_array[i] = i * (n / comm_sz);
      }
    }

    // calculate how many cards each process will handle
    double local_n = 0;
    if (my_rank == comm_sz - 1) {
        local_n = n / comm_sz + n % comm_sz;
    }
    else{
        local_n = n / comm_sz;
    }

    local_card_array = new MPI_Datatype[local_n];
    MPI_Bcast(&dimensions, 1, MPI_INT, 0, comm);

    // make sure all processes have the counts and displacements before scattering
    MPI_Barrier(comm); 

    // scatter the cards to all processes
    MPI_Scatterv(mpi_card_array, counts_array, dspls_array, mpi_type_card, local_card_array, local_n, mpi_type_card, 0, comm);
    

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        auto labels = kMeans(local_card_array, local_n, dimensions, k, iter, my_rank, comm);
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