import numpy as np
import chess
import chess.pgn
from typing import Optional, Tuple
from io import StringIO

def process_game(game: chess.pgn.Game) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a chess game into training data format.
    Returns two numpy arrays (white_moves, black_moves) of shape (num_moves, 34, 8, 8)
    Returns None if game has less than 10 moves per side
    """
    # Initialize board and move tracking
    board = chess.Board()
    positions_seen = set()  # Track positions for repetition detection
    positions_seen.add(board.fen().split(' ')[0])  # Add initial position
    
    # Lists to store move data
    white_moves = []
    black_moves = []
    
    # Process each move in the game
    node = game
    while node.next():
        node = node.next()
        move = node.move
        
        # Skip if move is None (incomplete game)
        if move is None:
            continue
            
        # Create planes for the current position
        current_planes = create_position_planes(board, positions_seen)
        
        # Make the move
        board.push(move)
        
        # Add new position to seen positions
        positions_seen.add(board.fen().split(' ')[0])
        
        # Create planes for the position after the move
        next_planes = create_position_planes(board, positions_seen)
        
        # Create the full 34-plane representation
        move_planes = np.zeros((34, 8, 8), dtype=np.float32)
        
        # First 13 planes (before move)
        move_planes[0:13] = current_planes
        
        # Next 13 planes (after move)
        move_planes[13:26] = next_planes
        
        # Castling availability (planes 27-30)
        move_planes[26] = float(board.has_queenside_castling_rights(chess.WHITE))
        move_planes[27] = float(board.has_kingside_castling_rights(chess.WHITE))
        move_planes[28] = float(board.has_queenside_castling_rights(chess.BLACK))
        move_planes[29] = float(board.has_kingside_castling_rights(chess.BLACK))
        
        # Side to move (plane 31)
        move_planes[30] = float(not board.turn)  # 0 for white, 1 for black
        
        # Fifty move counter (plane 32)
        move_planes[31] = board.halfmove_clock / 100.0
        
        # Move time - normalized between 0 and 1 (plane 33)
        # Extract clock info if available, otherwise use 0.5 as default
        clock_info = node.comment.strip('{}[] ').split()[1] if node.comment else "0:00:30"
        try:
            minutes, seconds = map(int, clock_info.split(':')[1:])
            total_seconds = minutes * 60 + seconds
            move_planes[32] = min(1.0, total_seconds / 180.0)  # Normalize to [0,1], capping at 3 minutes
        except:
            move_planes[32] = 0.5
        
        # All ones (plane 34)
        move_planes[33] = 1.0
        
        # Add to appropriate list based on whose move it was
        if board.turn:  # If it's Black's turn, it was White's move
            white_moves.append(move_planes)
        else:
            black_moves.append(move_planes)
    
    # Check if we have enough moves
    if len(white_moves) < 10 or len(black_moves) < 10:
        return None
    
    # Convert to numpy arrays
    white_array = np.stack(white_moves, axis=0)
    black_array = np.stack(black_moves, axis=0)
    
    return white_array, black_array

def create_position_planes(board: chess.Board, positions_seen: set) -> np.ndarray:
    """
    Create the 13 planes representing a board position.
    Uses efficient bitboard operations for piece placement.
    """
    planes = np.zeros((13, 8, 8), dtype=np.float32)
    
    # Piece placement planes (1-12)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # Process white pieces (planes 0-5)
    for i, piece_type in enumerate(piece_types):
        bb = board.pieces_mask(piece_type, chess.WHITE)
        planes[i] = bb_to_plane(bb)
    
    # Process black pieces (planes 6-11)
    for i, piece_type in enumerate(piece_types):
        bb = board.pieces_mask(piece_type, chess.BLACK)
        planes[i + 6] = bb_to_plane(bb)
    
    # Repetition plane (plane 12)
    current_position = board.fen().split(' ')[0]
    if list(positions_seen).count(current_position) > 1:
        planes[12] = 1.0
    
    return planes

def bb_to_plane(bb: int) -> np.ndarray:
    """
    Convert a bitboard to an 8x8 numpy array.
    Uses efficient bit operations.
    """
    # Convert bitboard to binary string and pad with zeros
    binary = format(bb, '064b')
    # Convert to numpy array and reshape
    return np.array([int(binary[i]) for i in range(64)], dtype=np.float32).reshape(8, 8)

def process_pgn_string(pgn_str: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Process a PGN string directly.
    Returns None if game is invalid or has less than 10 moves per side.
    """
    try:
        game = chess.pgn.read_game(StringIO(pgn_str))
        if game is None:
            return None
        return process_game(game)
    except:
        return None