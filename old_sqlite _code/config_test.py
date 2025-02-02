from torch import nn
def test_model_configuration(model):
    """Verify model configuration matches paper specifications"""
    print("\nTesting model configuration...")
    
    # Test MoveFeatureExtractor configuration
    assert len(model.move_extractor.blocks) == 6, "Should have 6 residual blocks"
    assert model.move_extractor.input_conv.out_channels == 64, "Channel size should be 64"
    assert model.move_extractor.final_proj.out_features == 320, "Move features should be 320-dimensional"
    
    # Test GameEncoder configuration
    encoder = model.game_encoder
    assert encoder.move_projection.in_features == 320, "Input dimension should match move features"
    assert encoder.move_projection.out_features == 1024, "Projection should be to 1024 dimensions"
    
    transformer = encoder.transformer
    assert len(transformer.layers) == 12, "Should have 12 transformer layers"
    assert transformer.layers[0].self_attn.num_heads == 8, "Should have 8 attention heads"
    
    final_proj = encoder.final_proj
    assert isinstance(final_proj, nn.Sequential), "Final projection should be sequential"
    assert final_proj[-1].out_features == 512, "Final embedding should be 512-dimensional"
    
    print("Model configuration matches paper specifications! ✓")