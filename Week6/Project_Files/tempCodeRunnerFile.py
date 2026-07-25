import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import time
import os

API_KEY = os.environ.get("TMDB_API_KEY", "18394374efe1306aed95ad29dccd67a5")
BASE_URL = "https://api.themoviedb.org/3"
NUM_PAGES = 50
OUT_PATH = "tmdb_raw_movies.csv"
REQUEST_DELAY = 0.3

def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Create a global session
session = create_session()

def get_genre_map():
    resp = session.get(
        f"{BASE_URL}/genre/movie/list",
        params={"api_key": API_KEY, "language": "en-US"},
        timeout=10
    )
    resp.raise_for_status()
    data = resp.json()["genres"]
    return {g["id"]: g["name"] for g in data}

def fetch_page(page: int):
    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": page,
        "include_adult": "false",
        "vote_count.gte": 20,
    }
    resp = session.get(f"{BASE_URL}/discover/movie", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", [])

def fetch_movie_details(movie_id: int):
    resp = session.get(
        f"{BASE_URL}/movie/{movie_id}",
        params={"api_key": API_KEY, "language": "en-US"},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()

def main():
    if API_KEY == "PASTE_YOUR_API_KEY_HERE" or API_KEY == "my api key":
        raise ValueError(
            "Set your TMDb API key first."
        )
    
    print("Fetching genre list...")
    genre_map = get_genre_map()
    print(f"Loaded {len(genre_map)} genres")
    
    all_movies = []
    for page in range(1, NUM_PAGES + 1):
        print(f"Fetching page {page}/{NUM_PAGES} ...")
        try:
            results = fetch_page(page)
            if not results:
                print("No more results, stopping.")
                break
            for m in results:
                m["genre_names"] = [
                    genre_map.get(gid, "Unknown")
                    for gid in m.get("genre_ids", [])
                ]
            all_movies.extend(results)
        except Exception as e:
            print(f"FAILED on page {page}: {e}")
            break
        time.sleep(REQUEST_DELAY)
    
    print(f"\nCollected {len(all_movies)} movies from discovery.")
    print("Fetching budget/runtime details...")
    
    detailed_rows = []
    for i, m in enumerate(all_movies):
        try:
            details = fetch_movie_details(m["id"])
            detailed_rows.append({
                "id": m["id"],
                "title": m.get("title"),
                "release_date": m.get("release_date"),
                "popularity": m.get("popularity"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "original_language": m.get("original_language"),
                "genres": "|".join(m.get("genre_names", [])),
                "budget": details.get("budget"),
                "revenue": details.get("revenue"),
                "runtime": details.get("runtime"),
            })
            if (i + 1) % 50 == 0:
                print(f"Details fetched: {i + 1}/{len(all_movies)}")
        except Exception as e:
            print(f"FAILED for movie {m.get('id')}: {e}")
        time.sleep(REQUEST_DELAY)
    
    df = pd.DataFrame(detailed_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} movies -> {OUT_PATH}")

if __name__ == "__main__":
    main()