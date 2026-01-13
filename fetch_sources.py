import csv
import yt_dlp
import time

def fetch_songs(queries, target_count=300):
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,  # Don't download, just extract metadata
        'force_generic_extractor': False,
        'noplaylist': True, # We want search results
    }

    unique_songs = {} # url -> {title, artist}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for query in queries:
            if len(unique_songs) >= target_count:
                break
            
            print(f"Searching for: {query}...")
            # Fetch 50 results per query
            try:
                # ytsearch50:query returns a plyalist-like object
                result = ydl.extract_info(f"ytsearch50:{query}", download=False)
                
                if 'entries' in result:
                    for entry in result['entries']:
                        if len(unique_songs) >= target_count:
                            break
                        
                        url = entry.get('url')
                        if url and url not in unique_songs:
                             # Construct full URL if it's just an ID (common in flat extraction)
                            if "youtube.com" not in url:
                                url = f"https://www.youtube.com/watch?v={url}"
                            
                            unique_songs[url] = {
                                "title": entry.get('title', 'Unknown Title'),
                                "artist": entry.get('uploader', 'Unknown Artist')
                            }
            except Exception as e:
                print(f"Error searching {query}: {e}")
            
    return unique_songs

def create_dataset():
    # 1. English Queries to get diverse 300 songs
    english_queries = [
        "top english songs 2024", "top english songs 2023", "top english songs 2022",
        "best pop songs 2020s", "billboard hot 100 hits", "classic rock hits",
        "top rap songs 2024", "best r&b songs", "taylor swift hits", "the weeknd hits"
    ]
    
    # 2. Bangla Queries to get diverse 300 songs
    bangla_queries = [
        "top bangla songs 2024", "best bangla songs 2023", "bangla romantic songs",
        "bangla band songs hits", "arijit singh bangla hits", "anupam roy hits",
        "rabindra sangeet best", "nazrul geeti best", "cokestudio bangla hits",
        "top bangla drama songs"
    ]

    print("Fetching English Songs...")
    english_songs = fetch_songs(english_queries, 300)
    print(f"Found {len(english_songs)} English songs.")

    print("Fetching Bangla Songs...")
    bangla_songs = fetch_songs(bangla_queries, 300)
    print(f"Found {len(bangla_songs)} Bangla songs.")

    # 3. Write to CSV
    header = ["language", "youtube_url", "title", "artist", "lyrics"]
    rows = []

    for url, data in english_songs.items():
        rows.append(["english", url, data['title'], data['artist'], ""])
        
    for url, data in bangla_songs.items():
        rows.append(["bangla", url, data['title'], data['artist'], ""])

    with open("youtube_sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Successfully created youtube_sources.csv with {len(rows)} total songs.")

if __name__ == "__main__":
    create_dataset()
