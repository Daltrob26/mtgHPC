#!/bin/bash
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N
#SBATCH --account=owner-guest
#SBATCH --partition=kingspeak-guest
#### IMPORTANT check which account and partition you can use
#### on the machine you are running on (you can use the 'myallocation' command)
cd ~/CS6030/mtgHPC
module load gcc cuda intel-mpi
#Run the program with our input
# nvcc kmeanChatGPT.cu -o cuda
# nvcc kmean.cu kmeanKernel.cu utils.cpp -o cuda
# ./cuda

# mpicc -c KMeansMPICuda.cpp -o main.o
# nvcc -c KMeansMPICuda.cu -o cuda_driver.o
# nvcc -c kmeanKernel.cu -o cuda_kmeans.o
# mpicc -c utils.cpp -o utils.o
# mpicc main.o cuda_driver.o cuda_kmeans.o utils.o -lcudart -lstdc++ -llzma -o mpi-cuda 
# srun --mpi=pmi2 ./mpi-cuda

mpicxx -c KMeansMPI.cpp -lcudart -o mpi
srun --mpi=pmi2 ./mpi