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


## Viz.py 

viz.py has a lot of dependencies, they're listed inside the pyproject.toml for the poetry enviornment you can use.

Running viz.py will launch a local webpage you and interact with the 2D visualisation of the clusted card data.

In particular it uses the file SerialCards.csv, if you want to look at the output from any other version of the program, just rename the files.

The visualization does pull data from scryfall to show the card that is currently being hovered over. We tried to be as nice to the api as possible so we cache the data in both memory while the program is running and store the data under assets/ . 

# Scaling Studies

## 1 vs 2

### Serial Performance

| Scale | Time (seconds) |
|-------|----------------|
| 1x    | 14.9481        |
| 2x    | 29.9797        |
| 5x    | 74.7923        |
| 10x   | 148.74         |

### OpenMP Performance

| Scale | Time (seconds) |
|-------|----------------|
| 1x    | 1.71803        |
| 2x    | 3.43699        |
| 5x    | 8.5891         |
| 10x   | 17.6898        |

![](/images/SerialVsOpenMP.png)

For the parallel memory implementation the most important step was dividing each cards distance computations across the processors. This was trivial in OpenMP. Recomputing the centroids location was a bit trickier. We gave each thread a local array to compute then put them back together in a critical section. While not the "most optimal" approach, this is sufficient for runtime. 

Assuming we get 1 core per task on CHPC this is our efficiency and speedup


![](/images/SpeedupOpenMP.png)
![](/images/EfficiencyOpenMP.png)

## 4 vs 5

### MPI Performance

| Scale | Time (seconds) |
|-------|----------------|
| 1x    | 1.05342        |
| 2x    | 3.43699        |
| 5x    | 8.5891         |
| 10x   | 17.6898        |

### Cuda-MPI Performance

| Scale | Time (seconds) |
|-------|----------------|
| 1x    | 1.71803        |
| 2x    | 3.43699        |
| 5x    | 8.5891         |
| 10x   | 17.6898        |


### Cuda

The average runtime of the cuda impelemtation on notchpeak across 10 runs for each block size :

| Block Size | Time (seconds) |
|------------|----------------|
| 64         | 0.568238       |
| 128        | 0.545512       |
| 256        | 0.543146       |
| 512        | 0.553101       |
| 1024       | 0.570377       |

We didn't observe a significant difference between block sizes until going up to 1024, which would be just one block per warp. The overall approach was similar to the other strategies, but the reduction at the end was much tricker due to not having any built in reduction tools. We did achieve a significant speedup compared the cpu impelemtations. The main notable part here is the cuda calculations were done in separate steps of assigning cards to centroids, computing new centroids, and doing a reduction to get the final result due to data dependencies.

### MPI

The average runtime of the MPI implementation on kingspeak across 10 runs for comm_sz count 2-8:

| Tasks | Time (seconds) |
|-------|----------------|
| 2     | 0.754343       |
| 4     | 0.389653       |
| 6     | 0.380434       |
| 8     | 0.296742       |

This implementation follows closesly to the sequence of events for the serial program with key differences at certain points. Reading from and writing to files is done only by process 0. To enable communication, the input data had to be translated to a data type that could be sent to different processes, so a CardMPI struct was made. The data was then scattered to all processes and the K-Means algorithm was run. In the algorithm, process 0 randomly chose and distributed the original centroids, then each process assigned their cards to a category. To recompute the centroids, each process summed up the feature vectors of their cards as well as how many cards were in each category, then those were both summed up via MPI_REDUCE functions. After this, process 0 calculated the new centroids based off of the average feature vectors, then broadcast this back to the other processes for the next cycle. Collecting the resultant label arrays was a simple process of having each thread send their label vectors to thread 0 where they were then added together into the resultant label vector.

### MPI Cuda

The average runtime of the cuda impelemtation on notchpeak across 10 runs for node counts 2-4:

| Nodes | Time (seconds) |
|-------|----------------|
| 2     | 0.394982       |
| 3     | 0.294263       |
| 4     | 0.235726       |

This impentation easily runs the fastest, but was fairly difficult and had more moving pieces. The biggest pain points were how to transfer the card data across nodes and keeping the two step reduction straight (one for cuda and one accross nodes). The cuda aspect was very similar to the first implementation and uses much of the same code. The MPI part first scatters the cards between the nodes, then for each iteration broadcasting the current centroids, performs the cuda computation on each node, then uses MPI_Allreduce and MPI_Gatherv to collect the results. The file loading and results output is only done on node 0.


# Credits
Dallin: serial, openMP, visualization, data prep
Logan: parallel CUDA GPU and distributed memory GPU
Matt: parallel MPI distributed memory
