# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech ESPnet model."""

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts_pretrain.abs_tts_pretrain import AbsTTSPretrain

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetTTSPretrainModel(AbsESPnetModel):
    """ESPnet model for text-to-speech pretraining task."""

    def __init__(
        self,
        tts_pretrain: AbsTTSPretrain,
    ):
        """Initialize ESPnetTTSModel module."""
        assert check_argument_types()
        super().__init__()
        self.tts_pretrain = tts_pretrain

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        lids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            kwargs: "utt_id" is among the input.

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.

        """
        # Make batch for tts inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
        )
        # Update batch for additional auxiliary inputs
        if lids is not None:
            batch.update(lids=lids)
        return self.tts_pretrain(**batch)

    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        return {}

