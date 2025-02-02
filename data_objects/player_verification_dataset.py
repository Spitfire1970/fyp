from data_objects.random_cycler import RandomCycler
from data_objects.player_batch import PlayerBatch
from data_objects.player import Player
from params_data import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import glob
import os

class PlayerVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path, games_per_player: int):
        self.root = datasets_root
        player_dirs = glob.glob(str(self.root) + "/**/*")
        if len(player_dirs) == 0:
            raise Exception("No players found. Make sure you are pointing to the directory "
                            "containing all preprocessed player directories.")

        self.players = [Player(Path(player_dir)) for player_dir in player_dirs]
        self.player_cycler = RandomCycler(self.players)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.player_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class PlayerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, players_per_batch, games_per_player, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.games_per_player = games_per_player

        super().__init__(
            dataset=dataset, 
            batch_size=players_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, players):
        random_partial_frames = np.random.randint(random_partial_low, random_partial_high+1)
        return PlayerBatch(players, self.games_per_player, random_partial_frames) 
    