#!/bin/bash

set -e

# Check for force flag to force compilation
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
if [ ! -f "mtg_features.csv" ]; then
    echo "mtg_features.csv not found. Generating..."
    python3 scryfallScraper.py
else
    echo "mtg_features.csv already exists"
fi

# Compile serial version
echo "Checking Serial executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "Serial" ]; then
    echo "Compiling serial version..."
    g++-15 KMeansSerial.cpp -o Serial
else
    echo "Serial already compiled"
fi

# Compile OpenMP version
echo "Checking OpenMP executable..."
if [ "$FORCE_COMPILE" = true ] || [ ! -f "OpenMP" ]; then
    echo "Compiling OpenMP version..."
    g++-15 -fopenmp KMeansOpenMP.cpp -o OpenMP
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
if cmp -s SerialCards.csv clusteredCards.csv; then
    echo "Files are identical"
else
    echo "Files are DIFFERENT"
    echo "Showing differences (first 20 lines):"
    diff SerialCards.csv clusteredCards.csv | head -n 20
fi

echo "Done."