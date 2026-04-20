#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -o slurmjob-%j.out-%N 
#SBATCH -e slurmjob-%j.err-%N 
#SBATCH --account=kingspeak-gpu 
#SBATCH --partition=kingspeak-gpu

module load gcc cuda intel-mpi

set -e

# Check for flag to force compilation
FORCE_COMPILE=false
if [ "$1" == "-f" ]; then
    FORCE_COMPILE=true
fi

# Check for cards.json
if [ ! -f "cards.json" ]; then
    echo "cards.json not found!"
    echo "Please download the Oracle cards bulk data from:"
    echo "https://scryfall.com/docs/api/bulk-data"
    echo "Save the file as 'cards.json' in this directory and rerun the script."
    exit 1
fi

# Check for mtg_features.csv
echo "Checking for mtg_features.csv..."
if [ ! -f "mtg_features.csv" ] || [ "$FORCE_COMPILE" = true ]; then
    echo "mtg_features.csv not found. Generating..."
    python3 scryfallScraper.py
else
    echo "mtg_features.csv already exists"
fi

# Compile serial version
echo "Checking Serial executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "Serial" ]; then
    echo "Compiling serial version..."
    g++ KMeansSerial.cpp  utils.cpp -o Serial
else
    echo "Serial already compiled"
fi

# Compile OpenMP version
echo "Checking OpenMP executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "OpenMP" ]; then
    echo "Compiling OpenMP version..."
    g++ -fopenmp KMeansOpenMP.cpp utils.cpp -o OpenMP
else
    echo "OpenMP already compiled"
fi

# Compile Cuda version
echo "Checking Cuda executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "Cuda" ]; then
    echo "Compiling Cuda version..."
    nvcc kmean.cu kmeanKernel.cu utils.cpp -o Cuda
else
    echo "Cuda already compiled"
fi

# Compile MPI version
echo "Checking MPI executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "MPI" ]; then
    echo "Compiling MPI version..."
    mpicxx KMeansMPI.cpp utils.cpp -lm -o  MPI
else
    echo "MPI already compiled"
fi

# Compile MPI Cuda version
echo "Checking Cuda executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "mpi-cuda" ]; then
    echo "Compiling MPI Cuda version..."
    mpicc -c KMeansMPICuda.cpp -o main.o
    nvcc -c KMeansMPICuda.cu -o cuda_driver.o
    nvcc -c kmeanKernel.cu -o cuda_kmeans.o
    mpicc -c utils.cpp -o utils.o
    mpicc main.o cuda_driver.o cuda_kmeans.o utils.o -lcudart -lstdc++ -llzma -o mpi-cuda 
else
    echo "MPI Cuda already compiled"
fi

compare_outputs () {
    echo "Comparing outputs for $1..."
    
    if [ "$1" == "OpenMP" ]; then
        TARGET="clusteredCardsOpenMP.csv"
    elif [ "$1" == "Cuda" ]; then
        TARGET="clusteredCardsCuda.csv"
    elif [ "$1" == "MPI" ]; then
        TARGET="clusteredCardsMPI.csv"
    elif [ "$1" == "MPICuda" ]; then
        TARGET="clusteredCardsMPICuda.csv"
    else
        echo "Unknown comparison target"
        return
    fi

    if cmp -s SerialCards.csv "$TARGET"; then
        echo "Files are identical"
    else
        echo "Files are DIFFERENT"
        echo "Showing differences (first 20 lines):"
        diff SerialCards.csv "$TARGET" | head -n 20
    fi
}

# Always run KMeans programs
echo -e "\nRunning serial version..."
./Serial

echo -e "\nRunning OpenMP version..."
./OpenMP
compare_outputs "OpenMP"

echo -e "\nRunning Cuda version..."
./Cuda
compare_outputs "Cuda"

echo -e "\nRunning MPI version..."
srun --mpi=pmi2 ./MPI
compare_outputs "MPI"

echo -e "\nRunning MPI Cuda version..."
srun --mpi=pmi2 ./mpi-cuda
compare_outputs "MPICuda"

echo "Done."