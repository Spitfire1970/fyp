import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeezed = self.squeeze(x).view(b, c)
        excited = self.excitation(squeezed).view(b, c, 1, 1)
        return x * excited

class ResidualBlock(nn.Module):
    """Residual block with SE module"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)

class MoveFeatureExtractor(nn.Module):
    """Extract features from chess moves using residual blocks"""
    def __init__(self, input_channels=34, num_blocks=6, channel_size=64):
        super().__init__()
        self.input_conv = nn.Conv2d(input_channels, channel_size, 3, padding=1)
        self.blocks = nn.ModuleList([
            ResidualBlock(channel_size) for _ in range(num_blocks)
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_proj = nn.Linear(channel_size, 320)

    def forward(self, x):

        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.final_proj(x)
        return x  

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GameEncoder(nn.Module):
    """Encode game-level features using Vision Transformer"""
    def __init__(self, d_model=1024, nhead=8, num_layers=12):
        super().__init__()
        self.move_projection = nn.Sequential(
            nn.Linear(320, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, 512)
        )

    def forward(self, move_features, mask=None):

        x = self.move_projection(move_features)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        x = self.norm(x)
        game_vector = self.final_proj(x)
        return game_vector  

class ChessStylometryModel(nn.Module):
    """Complete model for chess behavioral stylometry with improved normalization"""
    def __init__(self):
        super().__init__()
        self.move_extractor = MoveFeatureExtractor()
        self.game_encoder = GameEncoder()
        self.layer_norm = nn.LayerNorm(512)

    def forward(self, board_sequences, mask=None):

        _, seq_len = board_sequences.shape[:2]

        move_features = []
        for i in range(seq_len):
            features = self.move_extractor(board_sequences[:, i])
            move_features.append(features)
        move_features = torch.stack(move_features, dim=1)

        game_vector = self.game_encoder(move_features, mask)

        game_vector = self.layer_norm(game_vector)
        return game_vector

    def get_player_embedding(self, game_vectors):
        return game_vectors.mean(dim=0)

class GE2ELoss(nn.Module):
    """Generalized End-to-End Loss for player verification with numerical stability improvements"""
    def __init__(self, w=10.0, b=5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w))
        self.b = nn.Parameter(torch.tensor(b))

    def normalize_vectors(self, vectors):
        """Normalize vectors to unit length"""
        norm = torch.norm(vectors, dim=-1, keepdim=True)

        return vectors / (norm + 1e-8)

    def forward(self, game_vectors, player_indices):

        device = game_vectors.device
        player_indices = player_indices.to(device)

        unique_players = torch.unique(player_indices)
        N = len(unique_players)  

        games_per_player = [(player_indices == p).sum().item() for p in range(N)]
        M = min(games_per_player)  

        centroids = torch.zeros(N, M, game_vectors.shape[-1], device=device)

        for i in range(N):
            player_mask = (player_indices == i)
            player_games = game_vectors[player_mask]

            if len(player_games) > 0:

                player_games = player_games[:M]

                for j in range(M):
                    if len(player_games) > 1:

                        other_games = torch.cat([player_games[:j], player_games[j+1:]], dim=0)
                        centroid = other_games.mean(dim=0)
                    else:

                        centroid = player_games[0]

                    centroids[i, j] = self.normalize_vectors(centroid.unsqueeze(0)).squeeze(0)

        sim_matrix = torch.zeros(N, M, N, device=game_vectors.device)

        for i in range(N):
            for j in range(M):

                game_vector = game_vectors[i * M + j]

                sims = torch.sum(game_vector.unsqueeze(0) * centroids[:, j], dim=-1)

                print("SISMMSSM", sims)
                sim_matrix[i, j] = self.w * sims + self.b

        loss = 0
        for i in range(N):
            for j in range(M):
                correct_sim = sim_matrix[i, j, i]
                print("CORRECT SIM", correct_sim)
                exp_sum = torch.exp(sim_matrix[i, j]).sum()
                loss += -correct_sim + torch.log(exp_sum)

        avg_loss = loss / (N * M)
        return avg_loss