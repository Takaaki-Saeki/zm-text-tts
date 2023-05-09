import pytest
import torch

from espnet2.tts_pretrain.text_encoder import TextEncoderPretrain

@pytest.mark.parametrize(
    "langs", [1, 7]
)
@pytest.mark.parametrize(
    "adapter_type",
    ["residual", "transformer", "identity"]
)

def test_text_encoder(
    langs,
    adapter_type
):
    model = TextEncoderPretrain(
        idim=10,
        hidden_channels=4,
        text_encoder_attention_heads=2,
        text_encoder_ffn_expand=4,
        text_encoder_blocks=2,
        text_encoder_positionwise_layer_type="conv1d",
        text_encoder_positionwise_conv_kernel_size=1,
        text_encoder_positional_encoding_layer_type="rel_pos",
        text_encoder_self_attention_layer_type="rel_selfattn",
        text_encoder_activation_type="swish",
        text_encoder_normalize_before=True,
        text_encoder_dropout_rate=0.1,
        text_encoder_positional_dropout_rate=0.0,
        text_encoder_attention_dropout_rate=0.0,
        text_encoder_conformer_kernel_size=7,
        use_macaron_style_in_text_encoder=True,
        use_conformer_conv_in_text_encoder=True,
        # extra embedding related
        langs=langs,
        adapter_type=adapter_type,
    )
    inputs = dict(
        text=torch.randint(0, 10, (2, 4)),
        text_lengths=torch.tensor([4, 1], dtype=torch.long),
    )
    if langs > 0:
        inputs.update(lids=torch.randint(0, langs, (2, 1)))
    loss, *_ = model(**inputs)
    loss.backward()
