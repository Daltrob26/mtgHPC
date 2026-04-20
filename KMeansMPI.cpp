#include <stdio.h>
#include <mpi.h>     /* For MPI functions, etc */ 
#include "utils.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>

#define MAX_FEATURES 100
#define MAX_NAME_LEN 100

struct CardMPI {
    char name[MAX_NAME_LEN];
	int id;
    double features[MAX_FEATURES];
};

// write out the data, final col is the cluster the card belongs to
void writeCSVWithCardDataMPI(const std::string &filename,
                          const std::vector<Card> &data,
                          int* &labels,
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
double computeDistance(double* a,
                       double* b, int dimensions) {
  double sum = 0.0;
  for (int i = 0; i < dimensions; ++i) {
    double difference = a[i] - b[i];
    sum += difference * difference;
  }
  return std::sqrt(sum);
}

int* kMeans(CardMPI* local_card_array, double local_n, int dimensions, int k,
                        int max_iterations, int my_rank, double total_n, MPI_Comm comm) {

  int* labels = new int[int(total_n)];
  std::fill(labels, labels + int(total_n), 0);

  // seed the random starting cards to be the centroids
  //  08051993 because thats mtg's birthday
  double* mpi_centroids = new double[k * dimensions];
  std::mt19937 rng(851993);
  std::uniform_int_distribution<int> dist(0, local_n - 1   );

  if (my_rank == 0){
    for (int i = 0; i < k; ++i) {
      int index = dist(rng);
      mpi_centroids[i * dimensions] = local_card_array[index].features[0];
      for (int d = 1; d < dimensions; ++d) {
        mpi_centroids[i * dimensions + d] = local_card_array[index].features[d];
      }
    }
  }

  MPI_Bcast(mpi_centroids, k * dimensions, MPI_DOUBLE, 0, comm);
  
  for (int iterations = 0; iterations < max_iterations; iterations++){
    for (int cardNum = 0; cardNum < local_n; ++cardNum) {
      double shortestDistance = 1e18;
      int bestCluster = 0;
      for (int cluster = 0; cluster < k; ++cluster) {
        double distance = computeDistance(local_card_array[cardNum].features, &mpi_centroids[cluster * dimensions], dimensions);
        if (distance < shortestDistance) {
          shortestDistance = distance;
          bestCluster = cluster;
        }
      }
      labels[local_card_array[cardNum].id] = bestCluster;
    }
    // recompute centroids

    double* newCentroids = new double[k * dimensions]();
    int* counts = new int[k]();
    double* globalCentroids = new double[k * dimensions]();
    int* globalCounts = new int[k]();

    for (int i = 0; i < local_n; i++) {
      int cluster = labels[i];
      counts[cluster]++;

      for (int d = 0; d < dimensions; d++) {
        newCentroids[cluster * dimensions + d] += local_card_array[i].features[d];
      }
    }
    MPI_Reduce(newCentroids, globalCentroids, k * dimensions, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(counts, globalCounts, k, MPI_INT, MPI_SUM, 0, comm);
    if (my_rank == 0) {
      for (int c = 0; c < k; ++c) {
        if (globalCounts[c] == 0)
          continue;
        for (int d = 0; d < dimensions; ++d) {
          globalCentroids[c * dimensions + d] /= globalCounts[c];
        }
      }
      memcpy(mpi_centroids, globalCentroids, k * dimensions * sizeof(double));
    }
    MPI_Bcast(mpi_centroids, k * dimensions, MPI_DOUBLE, 0, comm);
    
  }

  return labels;
}

void Build_mpi_type(MPI_Datatype* mpi_card_type) {
  CardMPI dummy;

  MPI_Aint base_addr, id_addr, name_addr, features_addr;
  MPI_Get_address(&dummy, &base_addr);
  MPI_Get_address(&dummy.id, &id_addr);
  MPI_Get_address(&dummy.name, &name_addr);
  MPI_Get_address(&dummy.features, &features_addr);

  int array_of_blocklengths[3] = {MAX_NAME_LEN, 1, MAX_FEATURES};
  MPI_Datatype array_of_types[3] = {MPI_CHAR, MPI_INT, MPI_DOUBLE};

  MPI_Aint array_of_displacements[3] = {name_addr - base_addr, id_addr - base_addr, features_addr - base_addr};

  MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, mpi_card_type);
  MPI_Type_commit(mpi_card_type);


}

int main() {

    // MPI variables
    MPI_Comm comm;
    int my_rank, comm_sz;
    int* counts_array;
    int* dspls_array;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &my_rank);

    int k = 5;
    int iter = 100;
    int n = 0;

    double totalTime = 0.0;
    int dimensions;
    std::vector<std::string> header;

    std::vector<Card> data;
    CardMPI* mpi_card_array = nullptr;

    MPI_Datatype mpi_card_type;

    Build_mpi_type(&mpi_card_type);

    if (my_rank == 0) {
      std::ifstream infile("mtg_features.csv");
      std::string headerLine;
      std::getline(infile, headerLine);

      if (!headerLine.empty() &&
        (headerLine.back() == '\n' || headerLine.back() == '\r')) {
        headerLine.pop_back();
      }

      header = parseCSVRow(headerLine);

      // read in the data
      data = readCSV("mtg_features.csv");
      // convert the data to a format we can send with MPI
      n = data.size();
      dimensions = data[0].features.size();

      if (dimensions > MAX_FEATURES) {
            std::cerr << "ERROR: dimensions exceed MAX_FEATURES\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
      }

      mpi_card_array = new CardMPI[n];
      for (int i = 0; i < n; i++) {
        // copy name
        strncpy(
            mpi_card_array[i].name,
            data[i].name.c_str(),
            MAX_NAME_LEN
        );
		mpi_card_array[i].id = i;

        mpi_card_array[i].name[MAX_NAME_LEN - 1] = '\0';

        // copy features
        for (int d = 0; d < dimensions; d++) {
            mpi_card_array[i].features[d] = data[i].features[d];
        }

        // zero padding (important for deterministic MPI transfer)
        for (int d = dimensions; d < MAX_FEATURES; d++) {
            mpi_card_array[i].features[d] = 0.0;
        }
      }
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

    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // calculate how many cards each process will handle
    int local_n = 0;
    if (my_rank == comm_sz - 1) {
        local_n = n / comm_sz + n % comm_sz;
    }
    else{
        local_n = n / comm_sz;
    }

    CardMPI* local_card_array = new CardMPI[local_n];

    // make sure all processes have the counts and displacements before scattering
    MPI_Barrier(comm); 

    // scatter the cards to all processes
    MPI_Scatterv(mpi_card_array, counts_array, dspls_array, mpi_card_type, local_card_array, local_n, mpi_card_type, 0, comm);
    

    for (int run = 0; run < NUM_RUNS; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        int* labels = kMeans(local_card_array, local_n, dimensions, k, iter, my_rank, n, comm);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        totalTime += elapsed.count();
        std::cout << "Run " << run + 1 << " completed in " << elapsed.count() << " seconds.\n";
    }

    
    
    if (my_rank == 0) {
        double averageTime = totalTime / NUM_RUNS;
        std::cout << "Average time over " << NUM_RUNS << " runs: " << averageTime << " seconds.\n";
        
    }
    int* labels = kMeans(local_card_array, local_n, dimensions, k, iter, my_rank, n, comm);
	
	if (my_rank != 0){
		MPI_Send(labels, n, MPI_INT, 0, 0, comm);
	}
    
    
    if (my_rank == 0){
		int* global_labels = new int[n];
		int* recv_buffer = new int[n];
		memcpy(global_labels, labels, n * sizeof(int));
		for (int i = 1; i < comm_sz; i++) {
			MPI_Recv(recv_buffer, n, MPI_INT, i, 0, comm, MPI_STATUS_IGNORE);
			for (int j = 0; j < n; j++) {
				global_labels[j] += recv_buffer[j];
			}
		}
        // for (int i = 0; i < n; i++) {
        //     printf("Card %s assigned to cluster %d\n", mpi_card_array[i].name, global_labels[i]);
        // }
        // printf("test");
        writeCSVWithCardDataMPI("clusteredCardsMPI.csv", data, global_labels, header);
    }

    free(mpi_card_array);
    free(counts_array);
    free(dspls_array);
    free(local_card_array);
    free(labels);
    MPI_Type_free(&mpi_card_type);
    MPI_Finalize();

    return 0;
}