# MTG K Means Clustering

![](/images/viz.png)

scryfallScrapper.py searches through the scryfall bulk data found at https://scryfall.com/docs/api/bulk-data and extracts data on each card into a CSV.

Download the oracle cards bulk download and rename it to cards.json, place it in the same directory as scryfallScraper.py then run the program. 

The format of the exported CSV is as follows

CardName, Power, Toughness, ... etc. 

When doing K means clustering ensure you skip over the first column of data. 

BuildandRun.sh handles compilation, runtime and verification of all versions of the program. It treats the serial output as objective truth and compares the other progams to it.
-f to force recompilation of all programs

scryfallScraper.py has a global variable at the top to adjust the number of times to duplicate the data.


KMeans Implementations have a global variable to change the ammount of times they are ran to compute an average runtime. 

## Run cuda

```bash
module load gcc cuda
sbatch batch.sh
```

## Viz.py 

viz.py has a lot of dependencies, they're listed inside the pyproject.toml for the poetry enviornment you can use.

Running viz.py will launch a local webpage you and interact with the 2D visualisation of the clusted card data.

In particular it uses the file SerialCards.csv, if you want to look at the output from any other version of the program, just rename the files.

The visualization does pull data from scryfall to show the card that is currently being hovered over. We tried to be as nice to the api as possible so we cache the data in both memory while the program is running and store the data under assets/ . 

# Scaling Studies

## 1 vs 2

### Serial Performance

| Scale | Time (seconds) |
|-------|---------------|
| 1x    | 14.9481       |
| 2x    | 29.9797       |
| 5x    | 74.7923       |
| 10x   | 148.74        |

### OpenMP Performance

| Scale | Time (seconds) |
|-------|---------------|
| 1x    | 1.71803       |
| 2x    | 3.43699       |
| 5x    | 8.5891        |
| 10x   | 17.6898       |

![](/images/SerialVsOpenMP.png)

For the parallel memory implementation the most important step was dividing each cards distance computations across the processors. This was trivial in OpenMP. Recomputing the centroids location was a bit trickier. We gave each thread a local array to compute then put them back together in a critical section. While not the "most optimal" approach, this is sufficient for runtime. 

Assuming we get 1 core per task on CHPC this is our efficiency and speedup


![](/images/SpeedupOpenMP.png)
![](/images/EfficiencyOpenMP.png)
