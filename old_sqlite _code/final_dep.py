import sqlite3
import chess.pgn
import random
import hashlib
from pathlib import Path
from zstandard import ZstdDecompressor
from io import TextIOWrapper
import json

random.seed(88)

PGN_PATH_DIR = "pgn_files"
RATING_THRESHOLD = 2200
DB_PATH = "chess_games.db"
COMMIT_RATE = 10000
JSON_NAME = 'players_seen.json'

def create_database(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT,
        username_hash TEXT,
        color TEXT,
        moves TEXT,
        time_control TEXT,
        seen BOOLEAN,
        rating INTEGER,
        split TEXT,
        source_pgn TEXT,
        UNIQUE(game_id, username)
    )
    """)

    conn.commit()
    
    return conn

def clean_pgn(game):
    clean_game = chess.pgn.Game()
    node = game
    clean_node = clean_game
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move  
        if 'clk' in next_node.comment:
            clock_info = ''
            for i, part in enumerate(next_node.comment.split()):
                if part.startswith('[%clk') and i+1 in range(len(next_node.comment.split())):
                    clock_info = part + " "+next_node.comment.split()[i+1]
                    clean_node = clean_node.add_variation(move)
                    clean_node.comment = clock_info
                    break
            
        node = next_node
    
    return str(clean_game).split('\n\n')[1]

def hash_username(username: str) -> str:
    return hashlib.sha256(username.encode()).hexdigest()[:16]

def get_time_seconds(time_control: str) -> int:
    if not time_control or time_control == '-':
        return 0
    base_time = time_control.split('+')[0]
    return int(base_time) if base_time.isdigit() else 0

def process_pgn_file(pgn_path: str, conn: sqlite3.Connection, players_seen: dict):
    all_players = set(players_seen.keys())
    games_processed = 0
    with open(pgn_path, "rb") as compressed_file:
        with ZstdDecompressor().stream_reader(compressed_file) as reader:
            with TextIOWrapper(reader, encoding="utf-8") as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)

                    if game is None:
                        break
                    games_processed += 1
                    game_id = game.headers.get("Site", "").split('/')[-1]
                    print(games_processed)
                    white = hash_username(game.headers.get("White"))
                    black = hash_username(game.headers.get("Black"))
                    white_rating = int(game.headers.get("WhiteElo", "0"))
                    black_rating = int(game.headers.get("BlackElo", "0"))
                    time_control = game.headers.get("TimeControl", "")
                    moves = clean_pgn(game)
                    
                    if abs(white_rating - black_rating) > 300:
                        continue
                        
                    seconds = get_time_seconds(time_control)
                    if seconds <= 30 or seconds >= 3600:
                        continue
                    if seconds < 180: time_control = "bullet"
                    elif seconds < 600: time_control = "blitz"
                    elif seconds < 3600: time_control = "rapid"
                    
                    if white_rating >= RATING_THRESHOLD:
                        if white not in all_players:
                            all_players.add(white)
                            players_seen[white] = random.random() < 0.75
                        split = assign_split()
                        conn.execute("""
                            INSERT OR IGNORE INTO games 
                            (game_id, username_hash, color, 
                            moves, time_control, seen, rating, split, source_pgn)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (game_id, white, 'white',
                            moves, time_control, players_seen[white], white_rating, split, pgn_path))
                    
                    if black_rating >= RATING_THRESHOLD:
                        if black not in all_players:
                            all_players.add(black)
                            players_seen[black] = random.random() < 0.75
                        split = assign_split()
                        conn.execute("""
                            INSERT OR IGNORE INTO games 
                            (game_id, username_hash, color, 
                            moves, time_control, seen, rating, split, source_pgn)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (game_id, black, 'black',
                            moves, time_control, players_seen[black], black_rating, split, pgn_path))
                    if games_processed % COMMIT_RATE == 0:
                        conn.commit()
    
    conn.commit()
    pgn_file.close()

def assign_split() -> str:
    r = random.random()
    if r < 0.8:
        return "training"
    elif r < 0.9:
        return "reference"
    return "query"               

def main():
    players_seen = {}
    conn = create_database(DB_PATH)
    pgn_dir = Path(PGN_PATH_DIR)
    for pgn_file in pgn_dir.glob("*.pgn.zst"):
        process_pgn_file(str(pgn_file), conn, players_seen)
    print(len(players_seen))
    with open(JSON_NAME, 'w') as f:
        json.dump(players_seen, f)

if __name__ == "__main__":
    main()