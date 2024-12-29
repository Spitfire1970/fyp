import sqlite3
import chess.pgn
import random
import hashlib
from pathlib import Path
import io

random.seed(88)

PGN_PATH_DIR = "pgn_files"
RATING_THRESHOLD = 1000
DB_PATH = "chess_games.db"
COMMIT_RATE = 3

def create_database(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_offset INTEGER,
        source_pgn TEXT,
        username TEXT,
        username_hash TEXT,
        color TEXT,
        game_id TEXT,
        moves TEXT,
        time_control TEXT,
        seen BOOLEAN,
        rating INTEGER,
        split TEXT,
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
        
        clean_node = clean_node.add_variation(move)
        
        if 'clk' in next_node.comment:
            clock_info = ''
            for i, part in enumerate(next_node.comment.split()):
                if part.startswith('[%clk'):
                    clock_info = part + " "+next_node.comment.split()[i+1]
                    break
            clean_node.comment = clock_info
            
        node = next_node
    
    return str(clean_game).split('\n\n')[1]

def hash_username(username: str) -> str:
    return hashlib.sha256(username.encode()).hexdigest()[:16]

def get_time_seconds(time_control: str) -> int:
    if not time_control or time_control == '-':
        return 0
    base_time = time_control.split('+')[0]
    return int(base_time) if base_time.isdigit() else 0

def get_time_control(seconds):
    if seconds < 180: time_control = "bullet"
    elif seconds < 600: time_control = "blitz"
    elif seconds < 3600: time_control = "rapid"
    return time_control


def process_pgn_file(pgn_path: str, conn: sqlite3.Connection, players_seen: dict):
    all_players = set(players_seen.keys())
    pgn_file = open(pgn_path)
    games_processed = 0
    
    while True:
        offset = pgn_file.tell()
        game = chess.pgn.read_game(pgn_file)

        if game is None:
            break

        moves = clean_pgn(game)
        white = game.headers.get("White")
        black = game.headers.get("Black")
        white_rating = int(game.headers.get("WhiteElo", "0"))
        black_rating = int(game.headers.get("BlackElo", "0"))
        time_control = game.headers.get("TimeControl", "")
        game_id = game.headers.get("Site", "").split('/')[-1]
        
        if abs(white_rating - black_rating) > 300:
            continue
            
        seconds = get_time_seconds(time_control)
        time_control = get_time_control(seconds)
        if seconds <= 30 or seconds > 3600:
            continue
        
        if white_rating >= RATING_THRESHOLD:
            if white not in all_players:
                all_players.add(white)
                players_seen[white] = random.random() < 0.75
            split = assign_split()
            conn.execute("""
                INSERT OR IGNORE INTO games 
                (file_offset, source_pgn, username, username_hash, color, 
                game_id, moves, time_control, seen, rating, split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (offset, pgn_path, white, hash_username(white), 'white',
                  game_id, moves, time_control, players_seen[white], white_rating, split))
        
        if black_rating >= RATING_THRESHOLD:
            if black not in all_players:
                all_players.add(black)
                players_seen[black] = random.random() < 0.75
            split = assign_split()
            conn.execute("""
                INSERT OR IGNORE INTO games 
                (file_offset, source_pgn, username, username_hash, color, 
                game_id, moves, time_control, seen, rating, split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (offset, pgn_path, black, hash_username(black), 'black',
                  game_id, moves, time_control, players_seen[black], black_rating, split))
        games_processed += 1
        print(games_processed)
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
    for pgn_file in pgn_dir.glob("*.txt"):
        process_pgn_file(str(pgn_file), conn, players_seen)

if __name__ == "__main__":
    main()