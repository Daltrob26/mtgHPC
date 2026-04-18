#include <stdio.h>
#include <mpi.h>     /* For MPI functions, etc */ 
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include "utils.h"

#define MAX_FEATURES 57
#define MAX_NAME_LEN 100


void launch_cuda(
    const std::vector<Card>& data,
    const std::vector<double>& centroids,
    int k,
    int dim,
    std::vector<double>& local_sums,
    std::vector<int>& local_counts,
    std::vector<int>& labels
);

void cleanup_cuda();


struct CardMPI {
    char name[MAX_NAME_LEN];
    double features[MAX_FEATURES];
};


void Build_mpi_type(MPI_Datatype* mpi_type_card) {

    CardMPI dummy;

    MPI_Aint base, id_addr, name_addr, features_addr;

    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.name, &name_addr);
    MPI_Get_address(&dummy.features, &features_addr);

    int blocklens[2] = {
        MAX_NAME_LEN,
        MAX_FEATURES
    };

    MPI_Datatype types[2] = {
        MPI_CHAR,
        MPI_DOUBLE
    };

    MPI_Aint displs[2] = {
        name_addr - base,
        features_addr - base
    };

    MPI_Type_create_struct(
        2,
        blocklens,
        displs,
        types,
        mpi_type_card
    );

    MPI_Type_commit(mpi_type_card);
}

std::vector<Card> ConvertToOriginalCards(
    CardMPI* local_cards,
    int local_n,
    int dimensions
) {
    std::vector<Card> result;
    result.reserve(local_n);

    for (int i = 0; i < local_n; i++) {
        Card c;

        // reconstruct name
        c.name = std::string(local_cards[i].name);

        // reconstruct features (only up to real dimensions)
        c.features.assign(
            local_cards[i].features,
            local_cards[i].features + dimensions
        );

        result.push_back(std::move(c));
    }

    return result;
}


int main(void) {

    // MPI variables
    int comm_sz;
    int my_rank; 
    int* counts_array = nullptr;
    int* dspls_array = nullptr;
    int dimensions;
    int data_size;

    /* Start up MPI */
    MPI_Init(NULL, NULL); 

    /* Get the number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 

    /* Get my rank among all the processes */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

    /* Print my message */
    printf("Greetings from process %d of %d!\n", my_rank, comm_sz);

    std::vector<Card> data;
    CardMPI* mpi_card_array = nullptr;

    MPI_Datatype mpi_card_type;

    Build_mpi_type(&mpi_card_type);

    counts_array = new int[comm_sz];
    dspls_array = new int[comm_sz];

    std::vector<std::string> header;
    if (my_rank == 0) {
        // Load File
        std::ifstream infile("mtg_features.csv");
        std::string headerLine;
        std::getline(infile, headerLine);

        if (!headerLine.empty() &&
            (headerLine.back() == '\n' || headerLine.back() == '\r')) {
            headerLine.pop_back();
        }

        header = parseCSVRow(headerLine);

        int k = 5;
        int iter = 100;

        std::vector<int> labels;
        data = readCSV("mtg_features.csv");

        // convert the data to a format we can send with MPI
        data_size = data.size();
        dimensions = data[0].features.size();

        if (dimensions > MAX_FEATURES) {
            std::cerr << "ERROR: dimensions exceed MAX_FEATURES\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        mpi_card_array = new CardMPI[data_size];
        for (int i = 0; i < data_size; i++) {
            // copy name
            strncpy(
                mpi_card_array[i].name,
                data[i].name.c_str(),
                MAX_NAME_LEN
            );

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

        // calculate counts and displacements for scatterv
        int base = data_size / comm_sz;
        int rem  = data_size % comm_sz;

        int offset = 0;

        for (int i = 0; i < comm_sz; i++) {

            counts_array[i] = base + (i == comm_sz - 1 ? rem : 0);
            dspls_array[i] = offset;

            offset += counts_array[i];
        }
    }

    MPI_Bcast(&dimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // calculate how many cards each process will handle
    int base = data_size / comm_sz;
    int local_n = base;

    if (my_rank == comm_sz - 1) {
        local_n += data_size % comm_sz;
    }

    CardMPI* local_card_array = new CardMPI[local_n];

    // make sure all processes have the counts and displacements before scattering
    MPI_Barrier(MPI_COMM_WORLD); 

    // scatter the cards to all processes
    MPI_Scatterv(
        mpi_card_array,
        counts_array,
        dspls_array,
        mpi_card_type,
        local_card_array,
        local_n,
        mpi_card_type,
        0,
        MPI_COMM_WORLD
    );


    std::vector<Card> local_data = ConvertToOriginalCards(local_card_array, local_n, dimensions);

    int k = 5;
    int iter = 100;

    std::vector<double> local_sums(k * dimensions);
    std::vector<double> global_sums(k * dimensions);

    std::vector<int> local_counts(k);
    std::vector<int> global_counts(k);

    std::vector<int> local_labels;

    std::vector<int> labels;
    if (my_rank == 0) {
        labels.resize(data_size);
    }


    // Perform Computation 
    double totalTime = 0.0; 
    for (int run = 0; run < NUM_RUNS; ++run) { 
        auto start = std::chrono::high_resolution_clock::now(); 

        std::vector<double> centroids(k * dimensions, 0.0);

        if (my_rank == 0) {
            // seed the random starting cards to be the centroids
            //  08051993 because thats mtg's birthday
            std::mt19937 rng(851993);
            std::uniform_int_distribution<int> dist(0, data_size - 1);
            for (int c = 0; c < k; c++) {
                int idx = dist(rng);
                // print the names of cards used for centroids
                //  std::cout << data[idx].name << "\n";
                std::memcpy(&centroids[c * dimensions],
                            data[idx].features.data(),
                            dimensions * sizeof(double));
            }
        } else {
            centroids.resize(k * dimensions);
        }

        // main loop
        for (int it = 0; it < 100; it++) {
            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_counts.begin(), local_counts.end(), 0);
            
            MPI_Bcast(
                centroids.data(),
                k * dimensions,
                MPI_DOUBLE,
                0,
                MPI_COMM_WORLD
            );

            launch_cuda(
                local_data,
                centroids,
                k,
                dimensions,
                local_sums,
                local_counts,
                local_labels
            );

            MPI_Allreduce(
                local_sums.data(),
                global_sums.data(),
                k * dimensions,
                MPI_DOUBLE,
                MPI_SUM,
                MPI_COMM_WORLD
            );

            MPI_Allreduce(
                local_counts.data(),
                global_counts.data(),
                k,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD
            );

            for (int c = 0; c < k; c++) {
                if (global_counts[c] == 0) continue;

                for (int d = 0; d < dimensions; d++) {
                    centroids[c * dimensions + d] = global_sums[c * dimensions + d] / global_counts[c];
                }
            }

            MPI_Gatherv(
                local_labels.data(),   
                local_n,        
                MPI_INT, 
                labels.data(),  
                counts_array, 
                dspls_array,   
                MPI_INT, 
                0, 
                MPI_COMM_WORLD
            );
        }

        cleanup_cuda();
        
        auto end = std::chrono::high_resolution_clock::now(); 

        std::chrono::duration<double> elapsed = end - start; 
        totalTime += elapsed.count(); 
        
        // std::cout << "Run " << run + 1 << " completed in " 
        //         << elapsed.count() << " seconds.\n"; 
    } 

    double averageTime = totalTime / NUM_RUNS; 
    
    if (my_rank == 0) {
        std::cout << "Average time over " << NUM_RUNS 
                << ": " << averageTime << " seconds.\n"; 
        
        // Write Result 
        writeCSVWithCardData("clusteredCardsMPICuda.csv", data, labels, header);
    }

    /* Shut down MPI */
    MPI_Finalize(); 

    return 0;
} 
