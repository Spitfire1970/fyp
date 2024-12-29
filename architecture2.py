import sqlite3
import chess
import chess.pgn
import io
import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class GameData:
    game_id: str
    username: str
    color: str
    moves: List[str]
    time_control: str
    rating: int
    split: str

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, db_path: str, batch_size: int = 20, num_players: int = 40):
        self.db_path = db_path
        self.batch_size = batch_size
        self.num_players = num_players
        self.player_games = self._load_player_games()
        self.piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
    def _load_player_games(self) -> Dict[str, List[GameData]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get players with blitz games and seen=True
        query = """
        SELECT DISTINCT username 
        FROM games 
        WHERE time_control = 'blitz' AND seen = True
        GROUP BY username
        HAVING COUNT(*) >= 100
        LIMIT ?
        """
        cursor.execute(query, (self.num_players,))
        players = [row[0] for row in cursor.fetchall()]
        
        player_games = {}
        for player in players:
            query = """
            SELECT game_id, username, color, moves, time_control, rating, split
            FROM games
            WHERE username = ? AND time_control = 'blitz' AND seen = True
            LIMIT 100
            """
            cursor.execute(query, (player,))
            games = [GameData(*row) for row in cursor.fetchall()]
            player_games[player] = games
            
        conn.close()
        return player_games

    def _board_to_tensor(self, board: chess.Board, color: str) -> torch.Tensor:
        # Initialize 34-channel tensor (24 for pieces, 10 for metadata)
        tensor = torch.zeros(34, 8, 8)
        
        # Fill piece channels (first 24 channels, 12 for each position)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_idx = self.piece_to_index[piece.symbol()]
                tensor[piece_idx][rank][file] = 1
                # Duplicate piece information for second position
                tensor[piece_idx + 12][rank][file] = 1
        
        # Fill metadata channels (remaining 10 channels)
        # Channel 24: Repetition
        # Channel 25: Castling rights white kingside
        tensor[25].fill_(float(board.has_kingside_castling_rights(chess.WHITE)))
        # Channel 26: Castling rights white queenside
        tensor[26].fill_(float(board.has_queenside_castling_rights(chess.WHITE)))
        # Channel 27: Castling rights black kingside
        tensor[27].fill_(float(board.has_kingside_castling_rights(chess.BLACK)))
        # Channel 28: Castling rights black queenside
        tensor[28].fill_(float(board.has_queenside_castling_rights(chess.BLACK)))
        # Channel 29: Active color
        tensor[29].fill_(float(color == 'white'))
        # Channel 30: Fifty-move rule
        tensor[30].fill_(float(board.halfmove_clock) / 100.0)
        # Channel 31: Border
        tensor[31][0].fill_(1)
        tensor[31][7].fill_(1)
        tensor[31][:, 0].fill_(1)
        tensor[31][:, 7].fill_(1)
        # Channel 32-33: Move clock time (normalized)
        # Will be filled later when processing moves
        
        return tensor

    def _process_game(self, game_data: GameData) -> List[torch.Tensor]:
        board = chess.Board()
        tensors = []
        moves = []
        
        # Parse PGN
        pgn = io.StringIO(game_data.moves)
        game = chess.pgn.read_game(pgn)
        
        for node in game.mainline():
            move = node.move
            if move is None:
                continue
                
            # Only process moves by the target player
            is_white_to_move = board.turn == chess.WHITE
            if (is_white_to_move and game_data.color == 'white') or \
               (not is_white_to_move and game_data.color == 'black'):
                # Get clock time from comment if available
                clock_time = 60.0  # default 1 minute
                if node.comment and '[%clk' in node.comment:
                    time_str = node.comment.split('[%clk')[1].split(']')[0]
                    h, m, s = map(float, time_str.split(':'))
                    clock_time = h * 3600 + m * 60 + s
                
                # Create input tensor
                tensor = self._board_to_tensor(board, game_data.color)
                # Add clock time to metadata channels
                tensor[32].fill_(clock_time / 3600.0)  # Normalize to hours
                
                tensors.append(tensor)
                moves.append(move)
            
            board.push(move)
        
        return tensors

    def __len__(self) -> int:
        return len(self.player_games)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], str]:
        player = list(self.player_games.keys())[idx]
        games = self.player_games[player]
        
        # Process all games for the player
        all_tensors = []
        for game in games:
            game_tensors = self._process_game(game)
            all_tensors.extend(game_tensors)
        
        return all_tensors, player
    
dataset = ChessDataset("chess_games.db")