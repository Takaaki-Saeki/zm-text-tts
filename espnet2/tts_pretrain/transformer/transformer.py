# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-TTS related modules."""

import random
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.initialize import initialize
from espnet2.tts_pretrain.abs_tts_pretrain import AbsTTSPretrain
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)


class TransformerPretrain(AbsTTSPretrain):
    """Pretraining module for Transformer-TTS encoder.

    This is a module of masked language modeling pretraining for a Transformer-TTS encoder.
    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        embed_dim: int = 512,
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        elayers: int = 6,
        eunits: int = 1024,
        adim: int = 512,
        aheads: int = 4,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        eprenet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        # extra embedding related
        langs: Optional[int] = None,
        use_adapter: bool = False,
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
        self.use_scaled_pos_enc = use_scaled_pos_enc

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

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define transformer encoder
        if eprenet_conv_layers != 0:
            # encoder prenet
            encoder_input_layer = torch.nn.Sequential(
                EncoderPrenet(
                    idim=idim,
                    embed_dim=embed_dim,
                    elayers=0,
                    econv_layers=eprenet_conv_layers,
                    econv_chans=eprenet_conv_chans,
                    econv_filts=eprenet_conv_filts,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=eprenet_dropout_rate,
                    padding_idx=self.padding_idx,
                ),
                torch.nn.Linear(eprenet_conv_chans, adim),
            )
        else:
            encoder_input_layer = torch.nn.Embedding(
                num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
            )

        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        if self.langs:
            encoder_cls = EncoderWithLid
        else:
            encoder_cls = EncoderWithAdapter
        
        adapter = None
        if use_adapter:
            if adapter_type == "residual":
                adapter = ResidualAdapter(input_dim=adim, hidden_dim=256)
            elif adapter_type == "transformer":
                adapter = TransformerAdapter(input_dim=adim)
            elif adapter_type == "identity":
                adapter = IdentityAdapter()
            else:
                raise ValueError("adapter_type must be one of 'residual', 'transformer', 'identity")

        self.encoder = encoder_cls(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            adapter=adapter)

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha
        )

        self.mlm_head = BertLMPredictionHead(
            hidden_size=adim,
            vocab_size=idim)

    def _reset_parameters(self, init_type, init_enc_alpha=1.0):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)

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

        # report extra information
        if self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item()
            )

        return loss, stats, None

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        lids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # forward encoder
        # (B, T)
        x_masks = self._source_mask(ilens)
        xs_masked, mlm_target = self._bert_mask_v2(xs)
        # Injecting language embedding inside the encoder
        if self.langs:
            lid_embs = self.lid_emb(lids.view(-1))
            hs_mlm, _ = self.encoder(xs_masked, x_masks, lid_embs)
        else:
            hs_mlm, _ = self.encoder(xs_masked, x_masks)
        prediction_scores = self.mlm_head(hs_mlm)
        mlm_loss = self.mlm_criterion(
            prediction_scores.view(-1, self.idim),
            mlm_target.view(-1))
        mlm_acc = self.mlm_accuracy(
            prediction_scores.view(-1, self.idim),
            mlm_target.view(-1))

        return mlm_loss, mlm_acc

    def _source_mask(self, ilens):
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)


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


class EncoderWithAdapter(Encoder):
    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        
        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                xs, masks = self.adapter(xs, masks)
            else:
                xs = self.adapter(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    encoder_output = xs
                    # intermediate branches also require normalization.
                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)
                        xs = xs + self.conditioning_layer(intermediate_result)

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.

        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                xs, masks = self.adapter(xs, masks)
            else:
                xs = self.adapter(xs)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


class EncoderWithLid(Encoder):
    def forward(self, xs, masks, lang_emb=None):
        """Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).
            lang_emb (torch.Tensor): Language embedding (#batch, attention_dim).
        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).
        """
        if isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L),
        ):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        
        # Adding language embedding
        if lang_emb is not None:
            xs = xs + lang_emb.unsqueeze(1)
        
        # Adding adapter
        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                xs, masks = self.adapter(xs, masks)
            else:
                xs = self.adapter(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    encoder_output = xs
                    # intermediate branches also require normalization.
                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)
                        xs = xs + self.conditioning_layer(intermediate_result)

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None, lang_emb=None):
        """Encode input frame.
        Args:
            xs (torch.Tensor): Input tensor.
            masks (torch.Tensor): Mask tensor.
            cache (List[torch.Tensor]): List of cache tensors.
            lang_emb (torch.Tensor): Language embedding.
        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Mask tensor.
            List[torch.Tensor]: List of new cache tensors.
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        # Adding language embedding to the embedded frame
        if lang_emb:
            xs = xs + lang_emb.unsqueeze(1)

        # Adding adapter
        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                xs, masks = self.adapter(xs, masks)
            else:
                xs = self.adapter(xs)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


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
