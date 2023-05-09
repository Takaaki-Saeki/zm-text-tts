import pytest
import torch

from espnet2.tts_pretrain.transformer import TransformerPretrain


@pytest.mark.parametrize("eprenet_conv_layers", [0, 1])
@pytest.mark.parametrize("langs", [-1, 2])
@pytest.mark.parametrize(
    "use_adapter, adapter_type",
    [
        (False, "residual"),
        (True, "residual"),
        (True, "transformer"),
        (True, "identity")
    ]
)

def test_tranformer(
    eprenet_conv_layers,
    langs,
    use_adapter,
    adapter_type
):
    model = TransformerPretrain(
        idim=10,
        embed_dim=4,
        eprenet_conv_layers=eprenet_conv_layers,
        eprenet_conv_chans=256,
        eprenet_conv_filts=5,
        elayers=1,
        eunits=6,
        adim=4,
        aheads=2,
        positionwise_layer_type="conv1d",
        positionwise_conv_kernel_size=1,
        use_scaled_pos_enc=True,
        use_batch_norm=True,
        langs=langs,
        use_adapter=use_adapter,
        adapter_type=adapter_type
    )
    inputs = dict(
        text=torch.randint(0, 10, (2, 4)),
        text_lengths=torch.tensor([4, 1], dtype=torch.long),
    )
    if langs > 0:
        inputs.update(lids=torch.randint(0, langs, (2, 1)))
    loss, *_ = model(**inputs)
    loss.backward()
