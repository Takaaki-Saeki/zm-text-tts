# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-TTS related modules."""

import random
import math
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from typeguard import check_argument_types
from espnet2.tts_pretrain.abs_tts_pretrain import AbsTTSPretrain
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet2.gan_tts.vits.text_encoder import TextEncoder


class TextEncoderPretrain(AbsTTSPretrain):
    """Pretraining module for text encoder for VITS.

    This is a module of masked language modeling pretraining for a text encoder of VITS.
    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(
        self,
        idim: int,
        hidden_channels: int = 192,
        text_encoder_attention_heads: int = 2,
        text_encoder_ffn_expand: int = 4,
        text_encoder_blocks: int = 6,
        text_encoder_positionwise_layer_type: str = "conv1d",
        text_encoder_positionwise_conv_kernel_size: int = 1,
        text_encoder_positional_encoding_layer_type: str = "rel_pos",
        text_encoder_self_attention_layer_type: str = "rel_selfattn",
        text_encoder_activation_type: str = "swish",
        text_encoder_normalize_before: bool = True,
        text_encoder_dropout_rate: float = 0.1,
        text_encoder_positional_dropout_rate: float = 0.0,
        text_encoder_attention_dropout_rate: float = 0.0,
        text_encoder_conformer_kernel_size: int = 7,
        use_macaron_style_in_text_encoder: bool = True,
        use_conformer_conv_in_text_encoder: bool = True,
        # extra embedding related
        langs: Optional[int] = None,
        adapter_type: str = "residual",
    ):
        """Initialize Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            embed_dim (int): Dimension of character embedding.
            eprenet_conv_layers (int): Number of encoder prenet convolution layers.
            eprenet_conv_chans (int): Number of encoder prenet convolution channels.
            eprenet_conv_filts (int): Filter size of encoder prenet convolution.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            adim (int): Number of attention transformation dimensions.
            aheads (int): Number of heads for multi head attention.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            positionwise_layer_type (str): Position-wise operation type.
            positionwise_conv_kernel_size (int): Kernel size in position wise conv 1d.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            transformer_lr (float): Initial value of learning rate.
            transformer_warmup_steps (int): Optimizer warmup steps.
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_enc_dec_attn_dropout_rate (float): Dropout rate in source
                attention module.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            eprenet_dropout_rate (float): Dropout rate in encoder prenet.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim

        # use idx 0 as padding idx
        self.padding_idx = 0
        self.eos = idim - 1
        self.mask = 1 # same as unk id
        self.no_mask_tokens = [
            self.padding_idx,
            self.eos,
            self.mask
        ]
        self.masking_prob = 0.15
        self.randomize_prob = 0.10
        self.no_change_prob = 0.10
        self.mlm_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.mlm_accuracy = Accuracy(
            ignore_index=self.padding_idx)

        assert langs is not None, "Not supporting langs is None."
        assert langs > 0, "langs must be > 0 for pretraining."
        if adapter_type == "residual":
            adapter = ResidualAdapter(input_dim=hidden_channels, hidden_dim=128)
        elif adapter_type == "transformer":
            adapter = TransformerAdapter(input_dim=hidden_channels)
        elif adapter_type == "identity":
            adapter = IdentityAdapter()
        else:
            raise ValueError("adapter_type must be one of 'residual', 'transformer', 'identity")

        self.text_encoder = TextEncoderWithLid(
            vocabs=idim,
            attention_dim=hidden_channels,
            attention_heads=text_encoder_attention_heads,
            linear_units=hidden_channels * text_encoder_ffn_expand,
            blocks=text_encoder_blocks,
            positionwise_layer_type=text_encoder_positionwise_layer_type,
            positionwise_conv_kernel_size=text_encoder_positionwise_conv_kernel_size,
            positional_encoding_layer_type=text_encoder_positional_encoding_layer_type,
            self_attention_layer_type=text_encoder_self_attention_layer_type,
            activation_type=text_encoder_activation_type,
            normalize_before=text_encoder_normalize_before,
            dropout_rate=text_encoder_dropout_rate,
            positional_dropout_rate=text_encoder_positional_dropout_rate,
            attention_dropout_rate=text_encoder_attention_dropout_rate,
            conformer_kernel_size=text_encoder_conformer_kernel_size,
            use_macaron_style=use_macaron_style_in_text_encoder,
            use_conformer_conv=use_conformer_conv_in_text_encoder,
            langs=langs,
            adapter=adapter,
        )

        self.mlm_head = BertLMPredictionHead(
            hidden_size=hidden_channels,
            vocab_size=idim)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        lids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """

        text = text[:, : text_lengths.max()]  # for data-parallel

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        # calculate transformer outputs
        mlm_loss, mlm_acc = self._forward(
            xs=xs,
            ilens=ilens,
            lids=lids,
        )
        
        loss = mlm_loss

        stats = dict(
            mlm_loss=mlm_loss.item(),
        )
        if mlm_acc >= 0:
            stats.update(mlm_acc=mlm_acc.item())

        return loss, stats, None

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        lids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs_masked, mlm_target = self._bert_mask_v2(xs)
        # Injecting language embedding inside the encoder
        hs_mlm, _, _, _ = self.text_encoder(xs_masked, ilens, lids)
        # (B, C, T) --> (B, T, C)
        hs_mlm = hs_mlm.transpose(1, 2)
        prediction_scores = self.mlm_head(hs_mlm)
        mlm_loss = self.mlm_criterion(
            prediction_scores.view(-1, self.idim),
            mlm_target.view(-1))
        mlm_acc = self.mlm_accuracy(
            prediction_scores.view(-1, self.idim),
            mlm_target.view(-1))

        return mlm_loss, mlm_acc

    def _bert_mask_v2(self, xs):
        """Make masks for masked language modeling.

        Following png bert implementation, we use the following mask scheme:
            For 12% of tokens, we replace their token IDs with the mask id.
            For another 1.5% of tokens, we replace their token IDs with a random token.
            For another 1.5% of tokens, we do not perform masking or replacement.

        Args:
            xs (LongTensor): Batch of padded token ids (B, Tmax).
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked language modeling.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        full_mask = torch.rand(xs.shape, device=xs.device) < self.masking_prob
        for t in self.no_mask_tokens:
            full_mask &= xs != t
        unchanged = full_mask & (torch.rand(xs.shape, device=xs.device) < self.no_change_prob)
        random_token_mask = full_mask & (torch.rand(xs.shape, device=xs.device) < self.randomize_prob)
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        random_tokens = torch.randint(0, self.idim, (len(random_token_idx[0]),), device=xs.device)
        mask = full_mask & ~random_token_mask & ~unchanged
        xs_masked = xs.clone()
        mlm_target = xs.clone()
        xs_masked.masked_fill_(mask, self.mask)
        xs_masked[random_token_idx] = random_tokens
        mlm_target.masked_fill_(~full_mask, self.padding_idx)
        return xs_masked, mlm_target


class TextEncoderWithLid(TextEncoder):
    def __init__(
        self,
        vocabs: int,
        attention_dim: int = 192,
        attention_heads: int = 2,
        linear_units: int = 768,
        blocks: int = 6,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 3,
        positional_encoding_layer_type: str = "rel_pos",
        self_attention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        normalize_before: bool = True,
        use_macaron_style: bool = False,
        use_conformer_conv: bool = False,
        conformer_kernel_size: int = 7,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        langs: int = -1,
        adapter: Any = None,
    ):
        super().__init__(
            vocabs=vocabs,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            blocks=blocks,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            positional_encoding_layer_type=positional_encoding_layer_type,
            self_attention_layer_type=self_attention_layer_type,
            activation_type=activation_type,
            normalize_before=normalize_before,
            use_macaron_style=use_macaron_style,
            use_conformer_conv=use_conformer_conv,
            conformer_kernel_size=conformer_kernel_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
        )
        self.adapter = adapter
        self.lang_emb = torch.nn.Embedding(langs, attention_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        lids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input index tensor (B, T_text).
            x_lengths (Tensor): Length tensor (B,).

        Returns:
            Tensor: Encoded hidden representation (B, attention_dim, T_text).
            Tensor: Projected mean tensor (B, attention_dim, T_text).
            Tensor: Projected scale tensor (B, attention_dim, T_text).
            Tensor: Mask tensor for input tensor (B, 1, T_text).

        """
        x = self.emb(x) * math.sqrt(self.attention_dim)
        x_mask = (
            make_non_pad_mask(x_lengths)
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .unsqueeze(1)
        )
        # Adding language embedding
        x += self.lang_emb(lids.view(-1)).unsqueeze(1)
        # Applying adapter
        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                x, x_mask = self.adapter(x, x_mask)
            else:
                x = self.adapter(x)
        # encoder assume the channel last (B, T_text, attention_dim)
        # but mask shape shoud be (B, 1, T_text)
        x, _, = self.encoder(x, x_mask)

        # convert the channel first (B, attention_dim, T_text)
        x = x.transpose(1, 2)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)

        return x, m, logs, x_mask


class ResidualAdapter(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(input_dim, eps=1e-12)
        # Down projection
        self.in_linear = torch.nn.Linear(input_dim, hidden_dim)
        # Up projection
        self.out_linear = torch.nn.Linear(hidden_dim, input_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        origin = x
        x = self.in_linear(self.layer_norm(x))
        x = self.out_linear(self.relu(x))
        return x + origin


class TransformerAdapter(torch.nn.Module):
    def __init__(self, input_dim):
        """
        Adapter based on Transformer Encoder consisting of MultiHeadAttention.
        """
        super().__init__()
        self.transformer_encoder = EncoderLayer(
            input_dim,
            MultiHeadedAttention(4, input_dim, 0.1),
            PositionwiseFeedForward(
                input_dim, 2048, 0.1),
            dropout_rate=0.1,
            normalize_before=True,
            concat_after=False,
            stochastic_depth_rate=0.0)
    
    def forward(self, x, mask):
        # x: (batch, time, dim)
        return self.transformer_encoder(x, mask)


class IdentityAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Identity()
    def forward(self, x):
        return self.net(x)


class BertPredictionHeadTransform(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = self.gelu_new
        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=1e-12)

    def gelu_new(self, x):
        """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class Accuracy:

    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        pred = output.clone().argmax(dim=-1)
        mask = target == self.ignore_index
        pred.masked_fill_(mask, self.ignore_index)
        n_masked = mask.sum()
        n_correct = pred.eq(target).sum() - n_masked
        n_samples = len(target) - n_masked
        if n_samples == 0:
            return torch.tensor(-1.0).to(output.device)
        else:
            return n_correct / n_samples
