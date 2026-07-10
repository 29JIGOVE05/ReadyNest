
import requests
import pandas as pd
import time
import os

API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "I've removed my api key due security reasons")
 
MAX_REQUESTS = 950

AREAS = [
    "Burari, Delhi", "Karol Bagh, Delhi", "Ashok Vihar, Delhi", "Hauz Khas, Delhi",
    "Connaught Place, New Delhi", "Pitampura, Delhi", "Rajendra Nagar, Delhi",
    "Netaji Subhash Place, Delhi", "Rohini, Delhi", "Dwarka, Delhi",
    "Lajpat Nagar, Delhi", "Saket, Delhi", "Vasant Kunj, Delhi", "Janakpuri, Delhi",
    "Rajouri Garden, Delhi", "Paharganj, Delhi", "Chandni Chowk, Delhi",
    "Model Town, Delhi", "Shahdara, Delhi", "Mayur Vihar, Delhi",
    "Preet Vihar, Delhi", "Laxmi Nagar, Delhi", "Kalkaji, Delhi",
    "Greater Kailash, Delhi", "South Extension, Delhi", "Malviya Nagar, Delhi",
    "Munirka, Delhi", "Vikaspuri, Delhi", "Uttam Nagar, Delhi", "Tilak Nagar, Delhi",
    "Punjabi Bagh, Delhi", "Shalimar Bagh, Delhi", "Civil Lines, Delhi",
    "Kamla Nagar, Delhi", "Mukherjee Nagar, Delhi", "Patel Nagar, Delhi",
    "Moti Nagar, Delhi", "Okhla, Delhi", "Nehru Place, Delhi", "Mayapuri, Delhi",
]

CATEGORIES = [
    "restaurants", "cafes", "salons", "gyms", "clothing stores",
    "grocery stores", "electronics shops", "clinics", "bakeries",
    "pharmacies", "tutoring centers", "furniture stores", "mobile shops",
    "jewellery shops", "hardware stores",
]
 
OUTPUT_FILE = "delhi_business_raw.csv"

def search_places(query, api_key):
    """Calls Google Places API (New) - Text Search. Returns a list of places."""
    url = "https://places.googleapis.com/v1/places:searchText"
 
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join([
            "places.displayName",
            "places.formattedAddress",
            "places.rating",
            "places.userRatingCount",
            "places.websiteUri",
            "places.nationalPhoneNumber",
            "places.types",
            "places.businessStatus",
            "places.priceLevel",
        ])
    }
 
    body = {"textQuery": query, "maxResultCount": 20}
 
    response = requests.post(url, headers=headers, json=body)
 
    if response.status_code != 200:
        print(f"  ERROR ({response.status_code}) for query '{query}': {response.text[:200]}")
        return []
 
    return response.json().get("places", [])

def collect_all_data():
    all_rows = []
    request_count = 0
    total_planned = len(AREAS) * len(CATEGORIES)
 
    print(f"Planned queries: {total_planned} (areas={len(AREAS)} x categories={len(CATEGORIES)})")
    print(f"Safety cap: {MAX_REQUESTS} requests (stays under the 1,000/month free tier)\n")
 
    for area in AREAS:
        for category in CATEGORIES:
            if request_count >= MAX_REQUESTS:
                print(f"\n⚠️  Reached safety cap of {MAX_REQUESTS} requests. Stopping here to avoid charges.")
                return all_rows
 
            query = f"{category} in {area}"
            print(f"[{request_count + 1}/{min(total_planned, MAX_REQUESTS)}] {query}")
 
            places = search_places(query, API_KEY)
            request_count += 1
 
            for place in places:
                row = {
                    "area": area,
                    "category_searched": category,
                    "name": place.get("displayName", {}).get("text", ""),
                    "address": place.get("formattedAddress", ""),
                    "rating": place.get("rating", None),
                    "review_count": place.get("userRatingCount", None),
                    "website": place.get("websiteUri", ""),
                    "phone": place.get("nationalPhoneNumber", ""),
                    "types": ", ".join(place.get("types", [])),
                    "business_status": place.get("businessStatus", ""),
                    "price_level": place.get("priceLevel", ""),
                }
                all_rows.append(row)
 
            time.sleep(0.3)
 
    print(f"\nFinished all planned queries. Total requests used: {request_count}")
    return all_rows
 
 
if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("⚠️  Please paste your Google Places API key into API_KEY before running.")
    else:
        print("Starting data collection...\n")
        rows = collect_all_data()
 
        df = pd.DataFrame(rows)
        print(f"\nCollected {len(df)} raw rows (includes duplicates - script 2 will clean these).")
 
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved to {OUTPUT_FILE}")