import os
import chess.pgn
import random
import hashlib
from pathlib import Path
from zstandard import ZstdDecompressor
from io import TextIOWrapper
import json
from pgn_to_np import process_game
import numpy as np
import sys
import gzip

np.set_printoptions(threshold=sys.maxsize)

random.seed(88)

class StatsManager:
    def __init__(self, save_path: str = "stats.json"):
        self.save_path = save_path
        self.stats = {}
        self._load_stats()
        self.last_save = 0
        self.count = 0

    def __getitem__(self, key):
        return self.stats[key]
    
    def __setitem__(self, key, value):
        self.count += 1
        self.stats[key] = value
        if self.count - self.last_save > 100000:
            self.save()
            self.last_save = self.count
        
    def _load_stats(self):
        """Load stats from file if it exists, otherwise return empty dict."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    self.stats =  json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode {self.save_path}. Starting with empty stats.")
                self.stats = {}
    
    def __contains__(self, key):
        return key in self.stats
    
    def save(self) -> None:
        """Save current stats to file."""
        with open(self.save_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

class SavedDictWithSet:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data = {}
        self.keys_set = set()
        self.last_save = 0
        self.count = 0
        self.load()
    
    def load(self):
        """Load the dictionary from file if it exists"""
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.data = json.load(f)
                self.keys_set = set(self.data.keys())
    
    def save(self):
        """Save current state to file"""
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.count += 1
        self.data[key] = value
        self.keys_set.add(key)
        if self.count - self.last_save > 10000:
            self.save()
            self.last_save = self.count
    
    def __contains__(self, key):
        return key in self.keys_set
    
    def remove(self, key):
        if key in self.keys_set:
            del self.data[key]
            self.keys_set.remove(key)
            self.save()

def hash_username(username: str) -> str:
    return hashlib.sha256(username.encode()).hexdigest()[:16]

def get_time_seconds(time_control: str) -> int:
    if not time_control or time_control == '-':
        return 0
    base_time = time_control.split('+')[0]
    return int(base_time) if base_time.isdigit() else 0

def process_pgn_file(save_path, pgn_path, players_seen, stats):
    games_processed = 0
    with open(pgn_path, "rb") as compressed_file:
        with ZstdDecompressor().stream_reader(compressed_file) as reader:
            with TextIOWrapper(reader, encoding="utf-8") as pgn_file:
                if pgn_path in stats:
                    i = 0
                    while i <= stats[pgn_path]:
                        chess.pgn.read_game(pgn_file)
                        i+=1
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    stats[pgn_path] = stats.stats.get(pgn_path, 0) + 1
                    game_id = game.headers.get("Site", "").split('/')[-1]
                    white = hash_username(game.headers.get("White"))
                    black = hash_username(game.headers.get("Black"))
                    white_rating = int(game.headers.get("WhiteElo", "0"))
                    black_rating = int(game.headers.get("BlackElo", "0"))
                    seconds = get_time_seconds(game.headers.get("TimeControl", ""))
                    if seconds <= 30 or seconds >= 3600:
                        continue
                    if seconds < 180: time_control = "bullet"
                    elif seconds < 600: time_control = "blitz"
                    elif seconds < 3600: time_control = "rapid"

                    arrs = process_game(game)

                    if arrs is not None:
                        if white not in players_seen: players_seen[white] = "seen" if random.random() < 0.75 else "unseen"
                        if black not in players_seen: players_seen[black] = "seen" if random.random() < 0.75 else "unseen"

                        white_dir = os.path.join(save_path, time_control, players_seen[white], white, assign_split())
                        os.makedirs(white_dir, exist_ok=True)

                        f_name = os.path.join(white_dir, game_id + '.npy.gz')
                        with gzip.GzipFile(f_name, "w") as f:
                            np.save(f, arrs[0])

                        black_dir = os.path.join(save_path, time_control, players_seen[black], black, assign_split())
                        os.makedirs(black_dir, exist_ok=True)

                        f_name = os.path.join(black_dir, game_id + '.npy.gz')
                        with gzip.GzipFile(f_name, "w") as f:
                            np.save(f, arrs[1])
                    games_processed += 1
                    print(games_processed)
                    if games_processed:break
    pgn_file.close()

def assign_split() -> str:
    r = random.random()
    if r < 0.8:
        return "training"
    elif r < 0.9:
        return "reference"
    return "query"               

def main():
    pgn_dir = Path("data/pgn_files")
    players_seen = SavedDictWithSet('data/players_seen.json')
    stats = StatsManager('data/stats.json')
    for pgn_file in pgn_dir.glob("*.pgn.zst"):
        process_pgn_file("data/processed", str(pgn_file), players_seen, stats)

if __name__ == "__main__":
    main()