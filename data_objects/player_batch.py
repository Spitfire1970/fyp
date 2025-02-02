import numpy as np
from typing import List
from data_objects.player import Player

class PlayerBatch:
    def __init__(self, players: List[Player], games_per_player: int, n_frames: int):
        self.players = players
        self.partials = {p: p.random_partial(games_per_player, n_frames) for p in players}
        self.data = np.array([frames for p in players for _, frames, _ in self.partials[p]])
