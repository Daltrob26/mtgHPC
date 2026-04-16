#!/bin/bash
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#### IMPORTANT check which account and partition you can use
#### on the machine you are running on (you can use the 'myallocation' command)
cd ~/mtgHPC
module load gcc cuda intel-mpi
#Run the program with our input
# nvcc kmean.cu utils.cpp -o cuda
# nvcc kmeanChatGPT.cu -o cuda
# ./cuda

mpicc -c KMeansMPICuda.cpp -o main.o
nvcc -c KMeansMPICuda.cu -o cuda_main.o
mpicc -c utils.cpp -o utils.o
mpicc main.o cuda_main.o utils.o -lcudart -lstdc++ -llzma -o mpi-cuda 
srun --mpi=pmi2 ./mpi-cuda