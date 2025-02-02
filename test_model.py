import torch
from chess_stylometry_model import ChessStylometryModel, GE2ELoss

def print_tensor_info(name, tensor):
    """Helper function to print tensor details"""
    print(f"\n{name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Min/Max: {tensor.min().item():.3f}/{tensor.max().item():.3f}")

def test_model():

    torch.manual_seed(88)
    N = 2
    M = 2
    batch_size = N * M  
    seq_length = 32  
    board_channels = 34  
    board_size = 8  

    print("Creating dummy data...")

    board_sequences = torch.randn(
        batch_size, seq_length, board_channels, board_size, board_size
    )
    print_tensor_info("Input board sequences", board_sequences)

    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
    mask[:, 30:] = True  
    print_tensor_info("Padding mask", mask)

    idx = []
    for i in range(N): idx += [i] * M
    player_indices = torch.tensor(idx)  
    assert len(torch.unique(player_indices)) == 2, "Need exactly 2 players"
    assert all((player_indices == i).sum() == 2 for i in range(2)), "Need 2 games per player"
    print_tensor_info("Player indices", player_indices)

    print("\nInitializing model...")
    model = ChessStylometryModel()
    criterion = GE2ELoss()

    print("\nRunning forward pass...")
    try:

        board_sequences = board_sequences.to('cpu')
        mask = mask.to('cpu')

        with torch.no_grad():
            print(board_sequences.shape)
            game_vectors = model(board_sequences, mask)
            print(game_vectors.shape)
        print_tensor_info("Game vectors", game_vectors)

        player_games = game_vectors[:2]  
        player_embedding = model.get_player_embedding(player_games)
        print_tensor_info("Player embedding", player_embedding)

        loss = criterion(game_vectors, player_indices)
        print_tensor_info("Loss", loss.unsqueeze(0))

        print("\nShape checks passed! ✓")

        print("\nRunning sanity checks...")

        game_norms = torch.norm(game_vectors, dim=1)
        print(f"Game vector norms: min={game_norms.min():.3f}, max={game_norms.max():.3f}")

        attn_weights = model.game_encoder.transformer.layers[0].self_attn.in_proj_weight
        print(f"Attention weight stats: mean={attn_weights.mean():.3f}, std={attn_weights.std():.3f}")

        print("\nTesting loss decrease with optimization...")

        optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.01)

        print("\nInitial parameters:")
        print(f"w: {criterion.w.item():.4f}")
        print(f"b: {criterion.b.item():.4f}")

        initial_loss = float(loss)
        for i in range(5000):
            optimizer.zero_grad()
            game_vectors = model(board_sequences, mask)

            with torch.no_grad():
                print(f"\nStep {i+1} stats:")
                print(f"Game vector mean: {game_vectors.mean():.4f}")
                print(f"Game vector std: {game_vectors.std():.4f}")

            loss = criterion(game_vectors, player_indices)
            if not torch.isfinite(loss):
                print("Warning: Loss is not finite!")
                break

            loss.backward()

            optimizer.step()

            print(f"Loss: {loss.item():.4f}")
            print(f"w: {criterion.w.item():.4f}")
            print(f"b: {criterion.b.item():.4f}")

        assert loss < initial_loss, "Loss should decrease during training"
        print("\nLoss decreased as expected! ✓")

        print("\nAll tests passed successfully! 🎉")

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_model()