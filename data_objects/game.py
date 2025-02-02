from params_data import game_start
import gzip
import numpy as np


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