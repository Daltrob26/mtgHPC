# MTG K Means Clustering

scryfallScrapper.py searches through the scryfall bulk data found at https://scryfall.com/docs/api/bulk-data and extracts data on each card into a CSV.

Download the oracle cards bulk download and rename it to cards.json, place it in the same directory as scryfallScraper.py then run the program. 

BuildandRun.sh handles compilation, runtime and verification of all versions of the program. It treats the serial output as objective truth and compares the other progams to it.
-f to force recompilation of all programs

scryfallScraper.py has a global variable at the top to adjust the number of times to duplicate the data.

KMeansImplementations have a global variable to change the ammount of times they are ran to compute an average runtime. 

The format of the exported CSV is as follows

CardName, Power, Toughness, ... etc. 

When doing K means clustering ensure you skip over the first column of data. 

## Viz.py 

viz.py has a lot of dependencies, they're listed inside the pyproject.toml for the poetry enviornment you can use.

Running viz.py will launch a local webpage you and interact with the 2D visualisation of the clusted card data.

In particular it uses the file SerialCards.csv, if you want to look at the output from any other version of the program, just rename the files.

The visualization does pull data from scryfall to show the card that is currently being hovered over. We tried to be as nice to the api as possible so we cache the data in both memory while the program is running and store the data under assets/ . 