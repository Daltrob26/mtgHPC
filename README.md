# MTG K Means Clustering

scryfallScrapper.py searches through the scryfall bulk data found at https://scryfall.com/docs/api/bulk-data and extracts data on each card into a CSV.

scryfallScrapper.py expects a file named mtg_features.csv to exist in the same directory.
KmeanSerial.cpp expects a file named clusteredCards.csv to exist in the same directory. 

Download the oracle cards bulk download and rename it to cards.json, place it in the same directory as scryfallScraper.py then run the program. 

The format of the exported CSV is as follows

CardName, Power, Toughness, ... etc. 

When doing K means clustering ensure you skip over the first column of data. 


## CUDA

Run ```nvcc kmean.cu -o out && sbatch batch.sh``` to run the cuda implementation on a job.