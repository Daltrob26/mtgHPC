# MTG K Means Clustering

scryfallScrapper.py searches through the scryfall bulk data found at https://scryfall.com/docs/api/bulk-data and extracts data on each card into a CSV.

scryfallScrapper.py expects a file named mtg_features.csv to exist in the same directory.
KmeanSerial.cpp expects a file named clusteredCards.csv to exist in the same directory. 

Download the oracle cards bulk download and rename it to cards.json, place it in the same directory as scryfallScraper.py then run the program. 

The format of the exported CSV is as follows

CardName, Power, Toughness, ... etc. 

When doing K means clustering ensure you skip over the first column of data. 

## Viz.py 

viz.py has a lot of dependencies, they're listed inside the pyproject.toml for the poetry enviornment you have to use.

Running viz.py will launch a local webpage you and interact with the 2D visualisation of the clusted card data. 

The visualization does pull data from scryfall to show the card that is currently being hovered over. We tried to be as nice to the api as possible so we cache the data in both memory while the program is running and store the data under assets/ . 