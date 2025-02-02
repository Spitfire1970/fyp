from data_objects.random_cycler import RandomCycler
from data_objects.game import Game
from pathlib import Path

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
