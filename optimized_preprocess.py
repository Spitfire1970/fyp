import os
import chess.pgn
import random
import hashlib
from pathlib import Path
from zstandard import ZstdDecompressor
from io import TextIOWrapper
import json
import numpy as np
import sys
import gzip
from concurrent.futures import ProcessPoolExecutor
from pgn_to_np import process_game
import logging
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple
from functools import lru_cache

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

def assign_split() -> str:
    r = random.random()
    if r < 0.8:
        return "training"
    elif r < 0.9:
        return "reference"
    return "query"    

# Add LRU cache to frequently called functions
@lru_cache(maxsize=1024)
def hash_username(username: str) -> str:
    return hashlib.sha256(username.encode()).hexdigest()[:16]

@lru_cache(maxsize=128)
def get_time_seconds(time_control: str) -> int:
    if not time_control or time_control == '-':
        return 0
    base_time = time_control.split('+')[0]
    return int(base_time) if base_time.isdigit() else 0

# Optimize position planes creation with bitwise operations
def create_position_planes(board: chess.Board, positions_seen: set) -> np.ndarray:
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # Process all pieces in one pass
    for color, offset in [(chess.WHITE, 0), (chess.BLACK, 6)]:
        for i, piece_type in enumerate(piece_types):
            bb = board.pieces_mask(piece_type, color)
            if bb:  # Only process if pieces exist
                planes[i + offset] = np.frombuffer(
                    bb.to_bytes(8, byteorder='little'),
                    dtype=np.uint8
                ).reshape(8, 8).astype(np.float32)
    
    # Optimized repetition check
    current_position = board.fen().split(' ')[0]
    planes[12] = float(list(positions_seen).count(current_position) > 1)
    
    return planes

class BatchProcessor:
    def __init__(self, save_path: str, batch_size: int = 100):  # Reduced batch size
        self.save_path = save_path
        self.batch_size = batch_size
        self.white_batches: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        self.black_batches: Dict[str, List[Tuple[str, np.ndarray]]] = {}

    def add_game(self, time_control: str, player_type: str, player: str, 
                 game_id: str, arr: np.ndarray, color: str):
        key = f"{time_control}/{player_type}/{player}"
        batch = self.white_batches if color == 'white' else self.black_batches
        
        if key not in batch:
            batch[key] = []
        
        batch[key].append((game_id, arr))
        
        if len(batch[key]) >= self.batch_size:
            self.save_batch(key, color)

    def save_batch(self, key: str, color: str):
        batch = self.white_batches if color == 'white' else self.black_batches
        if key not in batch or not batch[key]:
            return
            
        logging.info(f"Saving batch of {len(batch[key])} games for {color} player in {key}")
            
        directory = os.path.join(self.save_path, key)
        os.makedirs(directory, exist_ok=True)
        
        for game_id, arr in batch[key]:
            f_name = os.path.join(directory, f"{game_id}.npy.gz")
            with gzip.GzipFile(f_name, "w") as f:
                np.save(f, arr)
        
        batch[key] = []

    def save_all(self):
        for key in self.white_batches:
            self.save_batch(key, 'white')
        for key in self.black_batches:
            self.save_batch(key, 'black')

def process_pgn_chunk(chunk_data: Tuple[str, int, int]) -> List[Dict]:
    pgn_path, start_pos, num_games = chunk_data
    processed_games = []
    games_read = 0
    logging.info(f"Processing chunk starting at position {start_pos} in {pgn_path}")
    
    # Add memory management
    import psutil
    process = psutil.Process()
    
    with open(pgn_path, "rb") as compressed_file:
        with ZstdDecompressor().stream_reader(compressed_file) as reader:
            with TextIOWrapper(reader, encoding="utf-8") as pgn_file:
                # Skip to start position
                for _ in range(start_pos):
                    chess.pgn.skip_game(pgn_file)
                
                for _ in range(num_games):
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                        
                    game_data = process_single_game(game)
                    if game_data:
                        processed_games.append(game_data)
    
    logging.info(f"Completed chunk from {pgn_path}, processed {len(processed_games)} games")
    return processed_games

def process_single_game(game: chess.pgn.Game) -> Dict:
    game_id = game.headers.get("Site", "").split('/')[-1]
    white = hash_username(game.headers.get("White"))
    black = hash_username(game.headers.get("Black"))
    seconds = get_time_seconds(game.headers.get("TimeControl", ""))
    
    if seconds <= 30 or seconds >= 3600:
        return None
        
    if seconds < 180: 
        time_control = "bullet"
    elif seconds < 600: 
        time_control = "blitz"
    else:
        time_control = "rapid"
    
    arrs = process_game(game)
    if arrs is None:
        return None
        
    return {
        'game_id': game_id,
        'white': white,
        'black': black,
        'time_control': time_control,
        'arrs': arrs
    }

def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging to both file and console
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(f'logs/processing_{timestamp}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def main():
    setup_logging()
    logging.info("Starting chess game processing")
    
    # Set lower number of worker processes based on available memory
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    max_workers = max(1, min(os.cpu_count(), int(total_memory_gb / 2)))
    logging.info(f"Using {max_workers} worker processes based on {total_memory_gb:.1f}GB total memory")
    pgn_dir = Path("data/pgn_files")
    players_seen = SavedDictWithSet('data/players_seen.json')
    stats = StatsManager('data/stats.json')
    batch_processor = BatchProcessor("data/processed", batch_size=1000)
    
    # Process files in parallel
    logging.info(f"Found {len(list(pgn_dir.glob('*.pgn.zst')))} PGN files to process")
    with ProcessPoolExecutor() as executor:
        total_processed = 0
        for pgn_file in tqdm(pgn_dir.glob("*.pgn.zst"), desc="Processing PGN files"):
            mem_usage = get_memory_usage()
            logging.info(f"Current memory usage: {mem_usage:.1f}MB")
            file_size = os.path.getsize(pgn_file)
            num_chunks = max(1, file_size // (100 * 1024 * 1024))  # 100MB chunks
            print(num_chunks)
            chunks = [(str(pgn_file), i * 1000, 1000) for i in range(14)]
            print(len(chunks))
            
            for processed_games in executor.map(process_pgn_chunk, chunks):
                for game_data in processed_games:
                    if not game_data:
                        continue
                        
                    # Assign seen/unseen status for new players
                    for player in (game_data['white'], game_data['black']):
                        if player not in players_seen:
                            players_seen[player] = "seen" if random.random() < 0.75 else "unseen"
                    
                    # Add games to batch processor
                    split = assign_split()
                    batch_processor.add_game(
                        game_data['time_control'],
                        players_seen[game_data['white']],
                        game_data['white'],
                        game_data['game_id'],
                        game_data['arrs'][0],
                        'white'
                    )
                    batch_processor.add_game(
                        game_data['time_control'],
                        players_seen[game_data['black']],
                        game_data['black'],
                        game_data['game_id'],
                        game_data['arrs'][1],
                        'black'
                    )
            
            batch_size = len(processed_games)
            total_processed += batch_size
            stats[str(pgn_file)] = stats.stats.get(str(pgn_file), 0) + batch_size
            logging.info(f"Progress: Processed {total_processed} games from {pgn_file}")
    
    # Save any remaining batches
    logging.info("Saving remaining batches...")
    batch_processor.save_all()
    logging.info("Processing completed successfully")

if __name__ == "__main__":
    main()