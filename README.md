# Movie Semantic Search Assignment

This repository contains my solution for the semantic search on movie plots assignment using **SentenceTransformers (all-MiniLM-L6-v2)**.

## Setup
1. Clone:
   ```bash
   git clone https://github.com/your-username/movie-search-assignment.git
   cd movie-search-assignment
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook:
   ```bash
   jupyter notebook movie_search_solution.ipynb
   ```

## Testing
Unit tests are provided in your classroom template repository under `test_movie_search.py`.  

```bash
python -m unittest test_movie_search.py -v
```

## Usage
Minimal example in Python:
```python
from movie_search import load_movies, build_encoder, compute_embeddings, search_movies

df = load_movies("movies.csv")
model = build_encoder()
embeddings = compute_embeddings(model, df["plot"])

results = search_movies("spy thriller in Paris", top_n=5, model=model, df=df, embeddings=embeddings)
print(results)
```
CLI example:
```bash
python movie_search.py --csv movies.csv --query "spy thriller in Paris" --top-n 5
```

here are my results :


============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-8.4.1, pluggy-1.6.0 -- C:\ai\1\movie-search-assignment\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\ai\1\movie-search-assignment
collecting ... collected 4 items

tests/test_movie_search.py::TestMovieSearch::test_search_movies_output_format PASSED [ 25%]
tests/test_movie_search.py::TestMovieSearch::test_search_movies_relevance PASSED [ 50%]
tests/test_movie_search.py::TestMovieSearch::test_search_movies_similarity_range PASSED [ 75%]
tests/test_movie_search.py::TestMovieSearch::test_search_movies_top_n PASSED [100%]

