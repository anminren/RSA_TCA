import torch
from transformers import BartConfig, BartModel

from models.bart_simple import BartSimple


def test_tiny_bart_forward_and_decoder_freeze(tmp_path):
    config = BartConfig(
        vocab_size=32,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=32,
    )
    BartModel(config).save_pretrained(tmp_path)

    model = BartSimple(
        model_name=str(tmp_path),
        n_channels=4,
        brain_head_dropout=0.0,
        freeze_decoder=True,
        output_hidden_states=False,
    )
    # BartBackbone deliberately exposes only _get_encoder; resolve the wrapped
    # decoder directly for the smoke assertion.
    wrapped_decoder = model.encoder.model.get_decoder()
    assert all(not parameter.requires_grad for parameter in wrapped_decoder.parameters())

    input_ids = torch.tensor([[2, 5, 6, 7, 1]])
    attention_mask = torch.ones_like(input_ids)
    target_word_mask = torch.tensor([[0, 0, 0, 1, 0]])
    output = model(input_ids, attention_mask, target_word_mask)

    assert output["brain_pred"].shape == (1, 4)
    output["brain_pred"].sum().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.encoder._get_encoder().parameters()
        if parameter.requires_grad
    )
