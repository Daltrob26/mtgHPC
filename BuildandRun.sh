#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH -o slurmjob-%j.out-%N 
#SBATCH -e slurmjob-%j.err-%N 
#SBATCH --account=kingspeak-gpu 
#SBATCH --partition=kingspeak-gpu

module load gcc

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
    g++ KMeansSerial.cpp -o Serial
else
    echo "Serial already compiled"
fi

# Compile OpenMP version
echo "Checking OpenMP executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "OpenMP" ]; then
    echo "Compiling OpenMP version..."
    g++ -fopenmp KMeansOpenMP.cpp -o OpenMP
else
    echo "OpenMP already compiled"
fi

# Always run KMeans programs
echo "Running serial version..."
./Serial

echo "Running OpenMP version..."
./OpenMP

# Compare outputs
echo "Comparing outputs..."
if cmp -s SerialCards.csv clusteredCardsOpenMP.csv; then
    echo "Files are identical"
else
    echo "Files are DIFFERENT"
    echo "Showing differences (first 20 lines):"
    diff SerialCards.csv clusteredCardsOpenMP.csv | head -n 20
fi

echo "Done."