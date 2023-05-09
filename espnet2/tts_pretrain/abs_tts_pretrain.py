# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class AbsTTSPretrain(torch.nn.Module, ABC):
    """TTS pretrining abstract class."""

    @abstractmethod
    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Calculate outputs and return the loss tensor."""
        raise NotImplementedError
