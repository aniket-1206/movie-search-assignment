import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load dataset ---
df = pd.DataFrame({
    'title': ['Spy Movie', 'Romance in Paris', 'Action Flick'],
    'plot': [
        'A spy navigates intrigue in Paris to stop a terrorist plot.',
        'A couple falls in love in Paris under romantic circumstances.',
        'A high-octane chase through New York with explosions.'
    ]
})

# --- Load model once (fast reuse) ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Precompute embeddings for all movie plots ---
embeddings = model.encode(df['plot'].tolist(), convert_to_tensor=False)

def search_movies(query, top_n=5):
    """Search for the most relevant movies given a query.
    Returns a DataFrame with: title, plot, similarity
    """
    query_embedding = model.encode([query], convert_to_tensor=False)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    df_with_sim = df.copy()
    df_with_sim['similarity'] = similarities

    return df_with_sim.sort_values(by='similarity', ascending=False).head(top_n)
