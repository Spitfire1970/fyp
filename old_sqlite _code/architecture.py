import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import chess
import chess.pgn
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import List, Tuple, Dict
import re

# Constants
NUM_RESIDUAL_BLOCKS = 6
CHANNEL_SIZE = 64
SE_REDUCTION = 8
NUM_TRANSFORMER_BLOCKS = 12
NUM_ATTENTION_HEADS = 8
MOVE_EMBEDDING_DIM = 1024
GAME_EMBEDDING_DIM = 512
MAX_MOVES = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MOMENTUM = 0.9

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, SE_REDUCTION)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out

class MoveFeatureExtractor(nn.Module):
    def __init__(self, in_channels=34):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, CHANNEL_SIZE, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(CHANNEL_SIZE)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(CHANNEL_SIZE) for _ in range(NUM_RESIDUAL_BLOCKS)
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(CHANNEL_SIZE, 320)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        for block in self.residual_blocks:
            x = block(x)
            
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerGameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.move_projection = nn.Linear(320, MOVE_EMBEDDING_DIM)
        self.pos_encoder = PositionalEncoding(MOVE_EMBEDDING_DIM)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MOVE_EMBEDDING_DIM,
            nhead=NUM_ATTENTION_HEADS,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=NUM_TRANSFORMER_BLOCKS
        )
        
        self.final_projection = nn.Sequential(
            nn.Linear(MOVE_EMBEDDING_DIM, GAME_EMBEDDING_DIM),
            nn.Tanh()
        )
        
    def forward(self, x, mask=None):
        x = self.move_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        # Average the move embeddings
        x = torch.mean(x, dim=1)
        x = self.final_projection(x)
        return x

class ChessStylometryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.move_extractor = MoveFeatureExtractor()
        self.game_encoder = TransformerGameEncoder()
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, game_sequences, sequence_lengths):
        batch_size, max_seq_len = game_sequences.shape[:2]
        
        # Process each move with the feature extractor
        move_features = []
        for i in range(max_seq_len):
            features = self.move_extractor(game_sequences[:, i])
            move_features.append(features)
        
        move_features = torch.stack(move_features, dim=1)
        
        # Create padding mask for transformer
        mask = torch.arange(max_seq_len)[None, :] >= sequence_lengths[:, None]
        mask = mask.to(game_sequences.device)
        
        # Get game embedding
        game_embedding = self.game_encoder(move_features, mask)
        return game_embedding

class ChessDataset(Dataset):
    def __init__(self, db_path: str, split: str = 'train'):
        self.db_path = db_path
        self.split = split
        self.games = self._load_games()
        self.player_to_idx = {player: idx for idx, player in enumerate(self.unique_players)}
        
    def _load_games(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT username, game_id, moves, color
            FROM games
            WHERE time_control = 'blitz' 
            AND seen = True 
            AND split = ?
        """
        
        cursor.execute(query, (self.split,))
        games = cursor.fetchall()
        conn.close()
        
        # Process and group games by player
        games_by_player = {}
        for username, game_id, moves, color in games:
            if username not in games_by_player:
                games_by_player[username] = []
            games_by_player[username].append((game_id, moves, color))
            
        self.unique_players = sorted(games_by_player.keys())
        
        # Format data for training
        formatted_games = []
        for player in self.unique_players:
            player_games = games_by_player[player]
            formatted_games.extend((player, game_data) for game_data in player_games)
            
        return formatted_games
    
    def _process_pgn(self, pgn_str: str, color: str) -> Tuple[torch.Tensor, int]:
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        board = game.board()
        
        moves = []
        move_times = []
        
        for node in game.mainline():
            # Extract move time from comment
            comment = node.comment
            if comment:
                time_match = re.search(r'\[%clk (\d+):(\d+):(\d+)\]', comment)
                if time_match:
                    h, m, s = map(int, time_match.groups())
                    total_seconds = h * 3600 + m * 60 + s
                    move_times.append(total_seconds)
                else:
                    move_times.append(0)
            else:
                move_times.append(0)
                
            moves.append(node.move)
            
        # Convert to tensor format (34 channels)
        tensor_moves = []
        for move, move_time in zip(moves, move_times):
            # First 24 channels for piece positions (12 piece types * 2 positions)
            tensor = torch.zeros(34, 8, 8)
            
            # Set piece positions
            board_tensor = torch.zeros(12, 8, 8)
            for square, piece in board.piece_map().items():
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
                board_tensor[piece_idx, rank, file] = 1
                
            tensor[:12] = board_tensor
            
            # Make the move and encode the resulting position
            board.push(move)
            board_tensor = torch.zeros(12, 8, 8)
            for square, piece in board.piece_map().items():
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_idx = (piece.piece_type - 1) + (6 if piece.color else 0)
                board_tensor[piece_idx, rank, file] = 1
            
            tensor[12:24] = board_tensor
            
            # Position metadata (10 channels)
            tensor[24] = 1 if board.is_repetition() else 0
            tensor[25] = 1 if board.has_kingside_castling_rights(True) else 0
            tensor[26] = 1 if board.has_queenside_castling_rights(True) else 0
            tensor[27] = 1 if board.has_kingside_castling_rights(False) else 0
            tensor[28] = 1 if board.has_queenside_castling_rights(False) else 0
            tensor[29] = 1 if color == 'white' else 0
            tensor[30] = board.halfmove_clock / 100.0  # Normalized
            tensor[31] = len(moves) / 200.0  # Normalized move number
            tensor[32] = move_time / 3600.0  # Normalized time (max 1 hour)
            tensor[33] = 1 if board.is_check() else 0
            
            tensor_moves.append(tensor)
            
        return torch.stack(tensor_moves), len(tensor_moves)
    
    def __len__(self):
        return len(self.games)
    
    def __getitem__(self, idx):
        player, (game_id, moves, color) = self.games[idx]
        move_sequence, seq_length = self._process_pgn(moves, color)
        
        return {
            'player_idx': self.player_to_idx[player],
            'game_id': game_id,
            'moves': move_sequence,
            'seq_length': seq_length
        }

def ge2e_loss(game_embeddings: torch.Tensor, player_indices: torch.Tensor, 
              w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Implements the Generalized End-to-End Loss for behavioral stylometry
    
    Args:
        game_embeddings: Shape (batch_size, embedding_dim)
        player_indices: Shape (batch_size,)
        w, b: Learnable scaling parameters
    """
    batch_size = game_embeddings.shape[0]
    unique_players = torch.unique(player_indices)
    num_players = len(unique_players)
    
    # Calculate centroids for each player
    centroids = []
    for player_idx in unique_players:
        player_mask = (player_indices == player_idx)
        player_embeddings = game_embeddings[player_mask]
        # Calculate centroid excluding the current game
        player_centroids = []
        for i in range(len(player_embeddings)):
            other_embeddings = torch.cat([
                player_embeddings[:i],
                player_embeddings[i+1:]
            ])
            centroid = torch.mean(other_embeddings, dim=0)
            player_centroids.append(centroid)
        centroids.extend(player_centroids)
    centroids = torch.stack(centroids)
    
    # Calculate similarities
    similarities = torch.zeros(batch_size, num_players)
    start_idx = 0
    for i, player_idx in enumerate(unique_players):
        player_mask = (player_indices == player_idx)
        num_games = player_mask.sum()
        
        # Calculate cosine similarities
        sims = F.cosine_similarity(
            game_embeddings[player_mask].unsqueeze(1),
            centroids[start_idx:start_idx + num_games].unsqueeze(0),
            dim=2
        )
        
        # Scale similarities
        sims = w * sims + b
        similarities[player_mask] = sims
        start_idx += num_games
    
    # Calculate loss
    loss = 0
    for i, player_idx in enumerate(unique_players):
        player_mask = (player_indices == player_idx)
        player_sims = similarities[player_mask]
        
        # Positive similarity is the diagonal
        pos_sims = torch.diag(player_sims)
        
        # Negative similarities are all off-diagonal elements
        neg_sims = player_sims.flatten()
        neg_sims = neg_sims[~torch.eye(len(player_sims), dtype=bool).flatten()]
        
        # Calculate loss for this player
        loss += -torch.mean(pos_sims) + torch.log(torch.sum(torch.exp(neg_sims)))
    
    return loss / num_players

def train_model(model: ChessStylometryModel, 
                train_loader: DataLoader,
                num_epochs: int,
                device: torch.device):
    """Training loop for the chess stylometry model"""
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=