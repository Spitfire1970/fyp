## Model parameters
residual_channels = 64
residual_blocks = 6
se_ratio = 8
seq_input_channels = 320 # input dimension to lstm/transformer
model_embedding_size = 512

## Training parameters
# learning_rate_init = 1e-4
learning_rate_init = 0.01
players_per_batch = 40
games_per_player = 20

v_players_per_batch = 40
v_games_per_player = 20
num_validate = 10

# random number of partial frames in a batch, need to double check the upper limit
random_partial_low = 32
random_partial_high = 32
game_start = 0

# 32 moves as a window, used for inference
partials_n_frames = 32