import os
import re
import time
import requests
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


CACHE_DIR = "assets"  # Dash serves files from /assets automatically
os.makedirs(CACHE_DIR, exist_ok=True)

# in-memory cache
image_cache = {}

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def get_cached_or_fetch_image(name):
    # Check memory cache first
    if name in image_cache:
        return image_cache[name]

    filename = sanitize_filename(name) + ".jpg"
    filepath = os.path.join(CACHE_DIR, filename)

    # Check disk cache
    if os.path.exists(filepath):
        path = f"/assets/{filename}"
        image_cache[name] = path
        return path

    # Fetch from Scryfall
    try:
        url = f"https://api.scryfall.com/cards/named?exact={name}"
        r = requests.get(url)

        if r.status_code == 200:
            data = r.json()

            img_url = None
            if 'image_uris' in data:
                img_url = data['image_uris'].get('normal')
            elif 'card_faces' in data:
                img_url = data['card_faces'][0]['image_uris'].get('normal')

            if img_url:
                img_data = requests.get(img_url).content

                with open(filepath, 'wb') as f:
                    f.write(img_data)

                path = f"/assets/{filename}"
                image_cache[name] = path

                
                time.sleep(0.05) 
                return path

    except Exception as e:
        print(f"Error fetching {name}: {e}")

    return ""


df = pd.read_csv("SerialCards.csv", quotechar='"')

# Ensure last column is cluster
cols = list(df.columns)
cols[-1] = 'cluster'
df.columns = cols

df['cluster'] = df['cluster'].astype(int)

#reduce to 2d
feature_cols = [c for c in df.columns if c not in ['name', 'cluster']]
features = df[feature_cols].values

pca = PCA(n_components=2)
coords = pca.fit_transform(features)

df['x'] = coords[:, 0]
df['y'] = coords[:, 1]

#dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("K-means MTG Cluster Viewer"),

    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_name='name',
            width=900,
            height=700
        )
    ),

    html.Div([
        html.Img(
            id='card-image',
            style={
                'maxWidth': '300px',
                'marginTop': '20px',
                'border': '1px solid #ccc'
            }
        )
    ])
])

#fetch image on hover
@app.callback(
    Output('card-image', 'src'),
    Input('scatter-plot', 'hoverData')
)
def update_image(hoverData):
    if hoverData is None:
        return ""

    name = hoverData['points'][0]['hovertext']
    return get_cached_or_fetch_image(name)

if __name__ == "__main__":
    app.run(debug=True)