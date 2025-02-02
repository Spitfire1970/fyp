from params import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import glob
import gzip
import numpy as np
import random

class RandomCycler:
    """
    Creates an internal copy of a sequence and allows access to its items in a constrained random 
    order. For a source sequence of n items and one or several consecutive queries of a total 
    of m items, the following guarantees hold (one implies the other):
        - Each item will be returned between m // n and ((m - 1) // n) + 1 times.
        - Between two appearances of the same item, there may be at most 2 * (n - 1) other items.
    """
    
    def __init__(self, source):
        if len(source) == 0:
            raise Exception("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int):
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out
    
    def __next__(self):
        return self.sample(1)[0]
    
class Game:
    def __init__(self, frames_fpath):
        self.frames_fpath = frames_fpath
        
    def get_frames(self):
        return np.load(gzip.GzipFile(self.frames_fpath, "r"))

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial game of n_frames
        
        :param n_frames: The number of frames of the partial game
        :return: the partial game frames and a tuple indicating the start and end of the 
        partial game in the complete game.
        """
        frames = self.get_frames()
        # if frames.shape[0] == n_frames:
        #     start = 0
        # else:
        try:
            boundary = frames.shape[0] - n_frames + 1
            min_length = game_start + n_frames
            if game_start >= boundary:
                # print("add paddings, game start {}, game length {}, num_frames {}".format(game_start, frames.shape[0], n_frames))
                frames = np.pad(frames, [(0, min_length - len(frames)), (0, 0), (0, 0), (0, 0)], "constant")
                boundary = frames.shape[0] - n_frames + 1

            start = np.random.randint(game_start, boundary)
        except Exception as e:
            print("=====")
            print(e, game_start, boundary, frames.shape, n_frames)
            exit(0)

        end = start + n_frames
        return frames[start:end], (start, end)

# Contains the set of games of a single player
class Player:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.games = None
        self.game_cycler = None
        # self.games = [Game(g) for g in self.root.iterdir() if g.suffix == '.gz']
        # self.game_cycler = RandomCycler(self.games)
        
    def _load_games(self):
        self.games = [Game(g) for g in self.root.iterdir() if g.suffix == '.gz']
        self.game_cycler = RandomCycler(self.games)
               
    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial games from the disk in a way that all 
        games come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial games to sample from the set of games from 
        that player. games are guaranteed not to be repeated if <count> is not larger than 
        the number of games available.
        :param n_frames: The number of frames in the partial game.
        :return: A list of tuples (game, frames, range) where game is an Game, 
        frames are the frames of the partial games and range is the range of the partial 
        game with regard to the complete game.
        """
        if self.games is None:
            self._load_games()

        games = self.game_cycler.sample(count)

        a = [(g,) + g.random_partial(n_frames) for g in games]

        return a

class PlayerBatch:
    def __init__(self, players: list[Player], games_per_player: int, n_frames: int):
        self.players = players
        self.partials = {p: p.random_partial(games_per_player, n_frames) for p in players}
        self.data = np.array([frames for p in players for _, frames, _ in self.partials[p]])

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