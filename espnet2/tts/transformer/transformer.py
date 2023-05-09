# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-TTS related modules."""

import math
import random
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet.nets.pytorch_backend.e2e_tts_transformer import (
    GuidedMultiHeadAttentionLoss,
    TransformerLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class Transformer(AbsTTS):
    """Transformer-TTS module.

    This is a module of text-to-speech Transformer described in `Neural Speech Synthesis
    with Transformer Network`_, which convert the sequence of tokens into the sequence
    of Mel-filterbanks.

    .. _`Neural Speech Synthesis with Transformer Network`:
        https://arxiv.org/pdf/1809.08895.pdf

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        embed_dim: int = 512,
        eprenet_conv_layers: int = 3,
        eprenet_conv_chans: int = 256,
        eprenet_conv_filts: int = 5,
        dprenet_layers: int = 2,
        dprenet_units: int = 256,
        elayers: int = 6,
        eunits: int = 1024,
        adim: int = 512,
        aheads: int = 4,
        dlayers: int = 6,
        dunits: int = 1024,
        postnet_layers: int = 5,
        postnet_chans: int = 256,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        reduction_factor: int = 1,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        lang_embed_dim: Optional[int] = None,
        lang_embed_integration_type: str = "add",
        use_gst: bool = False,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        transformer_enc_dec_attn_dropout_rate: float = 0.1,
        eprenet_dropout_rate: float = 0.5,
        dprenet_dropout_rate: float = 0.5,
        postnet_dropout_rate: float = 0.5,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
        bce_pos_weight: float = 5.0,
        loss_type: str = "L1",
        use_guided_attn_loss: bool = True,
        num_heads_applied_guided_attn: int = 2,
        num_layers_applied_guided_attn: int = 2,
        modules_applied_guided_attn: Sequence[str] = ("encoder-decoder"),
        guided_attn_loss_sigma: float = 0.4,
        guided_attn_loss_lambda: float = 1.0,
        use_mlm_loss: bool = False,
        mlm_loss_lambda: float = 1.0,
        lang_family_encoding: bool = False,
        num_lang_family: int = -1,
        holdout_lids: str = None,
        use_lid_loss: bool = False,
        lid_loss_level: str = "utterance",
        lid_loss_lambda: float = 1.0,
        use_adapter: bool = False,
        adapter_type: str = "residual",
        use_spk_adapter: bool = False,
        use_encoder_w_lid: bool = False,
    ):
        """Initialize Transformer module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            embed_dim (int): Dimension of character embedding.
            eprenet_conv_layers (int): Number of encoder prenet convolution layers.
            eprenet_conv_chans (int): Number of encoder prenet convolution channels.
            eprenet_conv_filts (int): Filter size of encoder prenet convolution.
            dprenet_layers (int): Number of decoder prenet layers.
            dprenet_units (int): Number of decoder prenet hidden units.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            adim (int): Number of attention transformation dimensions.
            aheads (int): Number of heads for multi head attention.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Filter size of postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            positionwise_layer_type (str): Position-wise operation type.
            positionwise_conv_kernel_size (int): Kernel size in position wise conv 1d.
            reduction_factor (int): Reduction factor.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type (str): How to integrate speaker embedding.
            use_gst (str): Whether to use global style token.
            gst_tokens (int): Number of GST embeddings.
            gst_heads (int): Number of heads in GST multihead attention.
            gst_conv_layers (int): Number of conv layers in GST.
            gst_conv_chans_list: (Sequence[int]): List of the number of channels of conv
                layers in GST.
            gst_conv_kernel_size (int): Kernel size of conv layers in GST.
            gst_conv_stride (int): Stride size of conv layers in GST.
            gst_gru_layers (int): Number of GRU layers in GST.
            gst_gru_units (int): Number of GRU units in GST.
            transformer_lr (float): Initial value of learning rate.
            transformer_warmup_steps (int): Optimizer warmup steps.
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            transformer_enc_dec_attn_dropout_rate (float): Dropout rate in source
                attention module.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            eprenet_dropout_rate (float): Dropout rate in encoder prenet.
            dprenet_dropout_rate (float): Dropout rate in decoder prenet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.
            bce_pos_weight (float): Positive sample weight in bce calculation
                (only for use_masking=true).
            loss_type (str): How to calculate loss.
            use_guided_attn_loss (bool): Whether to use guided attention loss.
            num_heads_applied_guided_attn (int): Number of heads in each layer to apply
                guided attention loss.
            num_layers_applied_guided_attn (int): Number of layers to apply guided
                attention loss.
            modules_applied_guided_attn (Sequence[str]): List of module names to apply
                guided attention loss.
            guided_attn_loss_sigma (float) Sigma in guided attention loss.
            guided_attn_loss_lambda (float): Lambda in guided attention loss.
            use_mlm_loss (bool): Whether to use masked language model loss.
            lang_family_encoding (bool): Whether to use language family encoding.
            num_lang_family (int): Number of language families.
            holdout_lids (str): List of language ids to holdout.
            use_lid_loss (bool): Whether to use language id loss.
        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.mask = 1 # same as unk id
        self.reduction_factor = reduction_factor
        self.use_gst = use_gst
        self.use_guided_attn_loss = use_guided_attn_loss
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_mlm_loss = use_mlm_loss
        self.mlm_loss_lambda = mlm_loss_lambda
        self.loss_type = loss_type
        self.use_guided_attn_loss = use_guided_attn_loss
        if self.use_guided_attn_loss:
            if num_layers_applied_guided_attn == -1:
                self.num_layers_applied_guided_attn = elayers
            else:
                self.num_layers_applied_guided_attn = num_layers_applied_guided_attn
            if num_heads_applied_guided_attn == -1:
                self.num_heads_applied_guided_attn = aheads
            else:
                self.num_heads_applied_guided_attn = num_heads_applied_guided_attn
            self.modules_applied_guided_attn = modules_applied_guided_attn
        self.use_lid_loss = use_lid_loss
        self.lid_loss_level = lid_loss_level
        self.lid_loss_lambda = lid_loss_lambda
        self.use_adapter = use_adapter

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        self.use_spk_adapter = use_spk_adapter
        if use_spk_adapter:
            self.spk_adapter = ResidualAdapter(input_dim=adim, hidden_dim=256)
        
        self.use_encoder_w_lid = use_encoder_w_lid

        if not lang_family_encoding:
            self.lang_family_encoding = lang_family_encoding
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
            adapter = None
            if use_adapter:
                adapter = None
                if adapter_type == "residual":
                    adapter = ResidualAdapter(input_dim=adim, hidden_dim=256)
                elif adapter_type == "transformer":
                    adapter = TransformerAdapter(input_dim=adim)
                elif adapter_type == "identity":
                    adapter = IdentityAdapter()
                else:
                    raise ValueError("adapter_type must be one of 'residual', 'transformer', 'identity")

            # define spk and lang embedding
            self.spks = None
            if spks is not None and spks > 1:
                self.spks = spks
                self.sid_emb = torch.nn.Embedding(spks, adim)
            
            self.langs = None
            if use_mlm_loss:
                self.langs = None
            elif use_lid_loss:
                self.langs = None
            elif use_encoder_w_lid:
                self.langs = None
                self.lid_emb = torch.nn.Embedding(langs, adim)
            elif langs is not None and langs > 1:
                self.langs = langs
                self.lid_emb = torch.nn.Embedding(langs, adim)

            if use_encoder_w_lid:
                encoder_cls = EncoderWithLid
            else:
                encoder_cls = EncoderWithAdapter

            # Not using EncoderLid.
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
        else:
            # Using separate encoder for language family
            self.lang_family_encoding = lang_family_encoding
            self.num_lang_family = num_lang_family
            encoder_list = []
            assert use_encoder_w_lid == False, "EncoderWithLid is not supported for lang-family encoding."
            for _ in range(num_lang_family):
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
                encoder_list.append(Encoder(
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
                    adapter=None))
            self.encoder = torch.nn.ModuleList(encoder_list)

            # Define spk embedding and not use language embedding
            self.spks = None
            if spks is not None and spks > 1:
                self.spks = spks
                self.sid_emb = torch.nn.Embedding(spks, adim)
            self.langs = None

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
            )

        # define projection layer
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # Define language embedding
        self.lang_embed_dim = None
        if lang_embed_dim is not None and lang_embed_dim > 0:
            self.lang_embed_dim = lang_embed_dim
            self.lang_embed_integration_type = lang_embed_integration_type
        if self.lang_embed_dim is not None:
            if self.lang_embed_integration_type == "add":
                self.lang_projection = torch.nn.Linear(self.lang_embed_dim, adim)
            else:
                self.lang_projection = torch.nn.Linear(adim + self.lang_embed_dim, adim)

        if self.use_mlm_loss:
            assert langs is not None, "langs must be specified when use_mlm_loss is True."
            assert lang_family_encoding is False, "Not supporting lang_family_encoding when use_mlm_loss is True."
            assert use_encoder_w_lid is False, "EncoderWithLid is not supported for lang-family encoding."
            self.mlm_head = BertLMPredictionHead(
                hidden_size=adim,
                vocab_size=idim)
            self.mlm_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)
            assert len(holdout_lids.strip().split()) > 0, "holdout_lids must be specified when use_mlm_loss is True."
            self.holdout_lids = torch.tensor(
                [int(x) for x in holdout_lids.strip().split()]
            )
            self.accuracy = Accuracy(
                ignore_index=self.padding_idx)
        
        if self.use_lid_loss:
            assert langs is not None, "langs must be specified when use_lid_loss is True."
            assert lang_family_encoding is False, "Not supporting lang_family_encoding when use_lid_loss is True."
            self.lid_net = LidPredictor(adim, langs, level=lid_loss_level)
            self.lid_criterion = torch.nn.CrossEntropyLoss()

        # define transformer decoder
        if dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                DecoderPrenet(
                    idim=odim,
                    n_layers=dprenet_layers,
                    n_units=dprenet_units,
                    dropout_rate=dprenet_dropout_rate,
                ),
                torch.nn.Linear(dprenet_units, adim),
            )
        else:
            decoder_input_layer = "linear"
        self.decoder = Decoder(
            odim=odim,  # odim is needed when no prenet is used
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            self_attention_dropout_rate=transformer_dec_attn_dropout_rate,
            src_attention_dropout_rate=transformer_enc_dec_attn_dropout_rate,
            input_layer=decoder_input_layer,
            use_output_layer=False,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)
        self.prob_out = torch.nn.Linear(adim, reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # define loss function
        self.criterion = TransformerLoss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )
        if self.use_guided_attn_loss:
            self.attn_criterion = GuidedMultiHeadAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
            )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

    def _reset_parameters(self, init_type, init_enc_alpha=1.0, init_dec_alpha=1.0):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            if not self.lang_family_encoding:
                self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            else:
                for i in range(self.num_lang_family):
                    idx = max(0, i-1)
                    self.encoder[idx].embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            feats (Tensor): Batch of padded target features (B, Lmax, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lembs (Optional[Tensor]): Batch of language embeddings (B, lang_embed_dim).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            joint_training (bool): Whether to perform joint training with vocoder.

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = feats
        olens = feats_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        if self.use_mlm_loss:
            batch_size_org = batch_size
            mlm_loss, mlm_acc, lid_loss_mlm = self._mlm_forward(xs=xs, ilens=ilens, lids=lids)
            mlm_loss *= self.mlm_loss_lambda
            lid_loss_mlm *= self.lid_loss_lambda
            (
                xs, ilens, ys, olens, spembs, sids,
                lembs, lids, labels, batch_size, is_empty
            ) = self._select_holdin_langs(
                xs=xs,
                ilens=ilens,
                ys=ys,
                olens=olens,
                spembs=spembs,
                sids=sids,
                lembs=lembs,
                lids=lids,
                labels=labels)
        else:
            mlm_loss = torch.tensor(0.).to(ys.device)
            mlm_acc = torch.tensor(0.).to(ys.device)
            lid_loss_mlm = torch.tensor(0.).to(ys.device)

        if self.use_mlm_loss and is_empty:
            loss = mlm_loss + lid_loss_mlm
            stats = dict(
                mlm_loss=mlm_loss.item(),
                mlm_acc=mlm_acc.item(),
                lid_loss_mlm=lid_loss_mlm.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size_org), loss.device
            )
            return loss, stats, weight

        # calculate transformer outputs
        after_outs, before_outs, logits, lid_loss = self._forward(
            xs=xs,
            ilens=ilens,
            ys=ys,
            olens=olens,
            spembs=spembs,
            sids=sids,
            lembs=lembs,
            lids=lids,
        )

        # modifiy mod part of groundtruth
        olens_in = olens
        if self.reduction_factor > 1:
            assert olens.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
            labels = labels[:, :max_olen]
            labels = torch.scatter(
                labels, 1, (olens - 1).unsqueeze(1), 1.0
            )  # see #3388

        # calculate loss values
        l1_loss, l2_loss, bce_loss = self.criterion(
            after_outs, before_outs, logits, ys, labels, olens
        )
        if self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = l2_loss + bce_loss
        elif self.loss_type == "L1+L2":
            loss = l1_loss + l2_loss + bce_loss
        else:
            raise ValueError("unknown --loss-type " + self.loss_type)
        
        # Adding lid loss
        loss += lid_loss
        # Adding mlm loss
        loss += mlm_loss
        # Adding lid loss with mlm
        loss += lid_loss_mlm

        stats = dict(
            l1_loss=l1_loss.item(),
            l2_loss=l2_loss.item(),
            bce_loss=bce_loss.item(),
            lid_loss=lid_loss.item(),
            mlm_loss=mlm_loss.item(),
            mlm_acc=mlm_acc.item(),
            lid_loss_mlm=lid_loss_mlm.item(),
        )

        # calculate guided attention loss
        if self.use_guided_attn_loss:
            # calculate for encoder
            if "encoder" in self.modules_applied_guided_attn:
                if not self.lang_family_encoding:
                    att_ws = []
                    for idx, layer_idx in enumerate(
                        reversed(range(len(self.encoder.encoders)))
                    ):
                        att_ws += [
                            self.encoder.encoders[layer_idx].self_attn.attn[
                                :, : self.num_heads_applied_guided_attn
                            ]
                        ]
                        if idx + 1 == self.num_layers_applied_guided_attn:
                            break
                    att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_text, T_text)
                    enc_attn_loss = self.attn_criterion(att_ws, ilens, ilens)
                else:
                    enc_attn_loss = torch.zeros_like(loss)
                    idxes = [int(x) for x in torch.unique(lids).tolist()]
                    # Merging lid=1 and lid=0 (lid=0 is unk)
                    idxes = list(set([max(0, x-1) for x in idxes]))
                    for lidx in idxes:
                        att_ws = []
                        for idx, layer_idx in enumerate(
                            reversed(range(len(self.encoder[lidx].encoders)))
                        ):
                            att_ws += [
                                self.encoder[lidx].encoders[layer_idx].self_attn.attn[
                                    :, : self.num_heads_applied_guided_attn
                                ]
                            ]
                            if idx + 1 == self.num_layers_applied_guided_attn:
                                break
                        att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_text, T_text)
                        enc_attn_loss += self.attn_criterion(att_ws, ilens, ilens)
                    enc_attn_loss /= len(idxes)
                loss = loss + enc_attn_loss
                stats.update(enc_attn_loss=enc_attn_loss.item())
            # calculate for decoder
            if "decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].self_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_feats, T_feats)
                dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
                loss = loss + dec_attn_loss
                stats.update(dec_attn_loss=dec_attn_loss.item())
            # calculate for encoder-decoder
            if "encoder-decoder" in self.modules_applied_guided_attn:
                att_ws = []
                for idx, layer_idx in enumerate(
                    reversed(range(len(self.decoder.decoders)))
                ):
                    att_ws += [
                        self.decoder.decoders[layer_idx].src_attn.attn[
                            :, : self.num_heads_applied_guided_attn
                        ]
                    ]
                    if idx + 1 == self.num_layers_applied_guided_attn:
                        break
                att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_feats, T_text)
                enc_dec_attn_loss = self.attn_criterion(att_ws, ilens, olens_in)
                loss = loss + enc_dec_attn_loss
                stats.update(enc_dec_attn_loss=enc_dec_attn_loss.item())

        # report extra information
        if self.use_scaled_pos_enc:
            if not self.lang_family_encoding:
                stats.update(
                    encoder_alpha=self.encoder.embed[-1].alpha.item(),
                    decoder_alpha=self.decoder.embed[-1].alpha.item(),
                )
            else:
                encoder_alpha = 0.
                idxes = [int(x) for x in torch.unique(lids).tolist()]
                # Merging lid=1 and lid=0 (lid=0 is unk)
                idxes = list(set([max(0, x-1) for x in idxes]))
                for lidx in idxes:
                    encoder_alpha += self.encoder[lidx].embed[-1].alpha.item()
                encoder_alpha /= len(idxes)
                stats.update(
                    encoder_alpha=encoder_alpha,
                    decoder_alpha=self.decoder.embed[-1].alpha.item(),
                )

        if not joint_training:
            stats.update(loss=loss.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), loss.device
            )
            return loss, stats, weight
        else:
            return loss, stats, after_outs
    
    def _mlm_forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        lids: Optional[torch.Tensor] = None):
        xs_mlm = xs.clone()
        ilens_mlm = ilens.clone()
        x_masks_mlm = self._source_mask(ilens_mlm)
        xs_masked_mlm, mlm_target = self._bert_mask(xs_mlm, ilens_mlm)
        hs_mlm, _ = self.encoder(xs_masked_mlm, x_masks_mlm)
        prediction_scores = self.mlm_head(hs_mlm)
        mlm_loss = self.mlm_criterion(
            prediction_scores.view(-1, self.idim),
            mlm_target.view(-1))
        mlm_acc = self.accuracy(
            prediction_scores.view(-1, self.idim),
            mlm_target.view(-1))
        if self.use_lid_loss:
            lid_loss_mlm = self._calc_lid_loss(hs_mlm, lids)
        else:
            lid_loss_mlm = torch.tensor(0.).to(xs.device)
        return mlm_loss, mlm_acc, lid_loss_mlm

    def _calc_lid_loss(self, hs, lids):
        predicted = self.lid_net(hs)
        if self.lid_loss_level == "utterance":
            if len(lids.shape) == 2:
                target = lids.squeeze(1)
            else:
                target = lids
            lid_loss = self.lid_criterion(predicted, target)
        elif self.lid_loss_level == "token":
            predicted = predicted.transpose(1, 2) # (batch, n_langs, seq_len)
            if len(lids.shape) == 2:
                target = lids.repeat(1, predicted.size(2))
            else:
                target = lids.unsqueeze(1).repeat(1, predicted.size(2))
            lid_loss = self.lid_criterion(predicted, target)
        return lid_loss

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        spembs: torch.Tensor,
        sids: torch.Tensor,
        lembs: torch.Tensor,
        lids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # forward encoder
        x_masks = self._source_mask(ilens)

        if self.lang_family_encoding:
            hs, h_masks = self._multiple_encoding(xs, x_masks, lids)
        elif self.use_encoder_w_lid:
            lid_embs = self.lid_emb(lids.view(-1))
            hs, h_masks = self.encoder(xs, x_masks, lid_embs)
        else:
            hs, h_masks = self.encoder(xs, x_masks)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        
        if self.lang_embed_dim is not None:
            hs = self._integrate_with_lang_embed(hs, lembs)
        
        # Using speaker adapter
        if self.use_spk_adapter:
            hs = self.spk_adapter(hs)

        # Calculating lid loss inside _forward() if not using mlm loss
        # If using mlm loss, lid loss is calculated in _mlm_forward()
        if self.use_lid_loss and not self.use_mlm_loss:
            lid_loss = self._calc_lid_loss(hs, lids)
            lid_loss *= self.lid_loss_lambda
        else:
            lid_loss = torch.tensor(0.).to(xs.device)

        # thin out frames for reduction factor
        # (B, T_feats, odim) ->  (B, T_feats//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1 :: self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, h_masks)
        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, T_feats//r, r) -> (B, T_feats//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs, before_outs, logits, lid_loss

    def _select_holdin_langs(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor,
        olens: torch.Tensor,
        spembs: torch.Tensor,
        sids: torch.Tensor,
        lembs: torch.Tensor,
        lids: torch.Tensor,
        labels: torch.Tensor):
        # Selecting batch index corresponding to hold-in languages
        if len(lids.shape) == 1:
            lids_tmp = lids.unsqueeze(1)
        else:
            lids_tmp = lids
        cond = torch.any((lids_tmp == self.holdout_lids.to(lids.device)), dim=1)
        holdin_idx = torch.argwhere(~cond).squeeze()
        is_empty = False
        out = []
        batch_size = None
        for item in (xs, ilens, ys, olens, spembs, sids, lembs, lids):
            if len(holdin_idx.shape) == 0:
                if item is not None:
                    item = item[holdin_idx].unsqueeze(0)
                    batch_size = 1
            elif len(holdin_idx) == 0:
                is_empty = True
                batch_size = 0
            elif item is not None:
                item = item[holdin_idx]
                batch_size = len(holdin_idx)
            out += [item,]
        # Length of xs
        xs, ilens, ys, olens = out[:4]
        xs = xs[:, :ilens.max()]
        ys = ys[:, :olens.max()]
        # Recreate labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)
        if not is_empty:
            assert batch_size == xs.shape[0]
        return (xs, ilens, ys, olens, out[4], out[5], out[6], out[7], labels, batch_size, is_empty)

    def _multiple_encoding(
        self, xs: torch.Tensor, x_masks: torch.Tensor, lids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # forward encoder
        idxes = [int(x) for x in torch.unique(lids).tolist()]
        # Merging lid=1 and lid=0 (lid=0 is unk)
        idxes = list(set([max(0, x-1) for x in idxes]))
        for i, idx in enumerate(idxes):
            if i == 0:
                hs_tmp, h_masks = self.encoder[idx](xs, x_masks)
                hs_out = torch.zeros_like(hs_tmp)
            else:
                hs_tmp, _ = self.encoder[idx](xs, x_masks)
            mask = (lids == idx).unsqueeze(-1)
            hs_out += (hs_tmp * mask)
        return hs_out, h_masks

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor]): Feature sequence to extract style embedding
                (T_feats', idim).
            spembs (Optional[Tensor]): Speaker embedding (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lembs (Optional[Tensor]): Language embedding (lang_embed_dim,).
            lids (Optional[Tensor]): Language ID (1,).
            threshold (float): Threshold in inference.
            minlenratio (float): Minimum length ratio in inference.
            maxlenratio (float): Maximum length ratio in inference.
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * prob (Tensor): Output sequence of stop probabilities (T_feats,).
                * att_w (Tensor): Source attn weight (#layers, #heads, T_feats, T_text).

        """
        x = text
        y = feats
        spemb = spembs
        lemb = lembs

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert feats is not None, "feats must be provided with teacher forcing."

            # get teacher forcing outputs
            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            spembs = None if spemb is None else spemb.unsqueeze(0)
            lembs = None if lemb is None else lemb.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, *_ = self._forward(
                xs=xs,
                ilens=ilens,
                ys=ys,
                olens=olens,
                spembs=spembs,
                sids=sids,
                lembs=lembs,
                lids=lids,
            )
            if outs is None:
                return dict(feats_gen=None, att_w=None)

            # get attention weights
            att_ws = []
            for i in range(len(self.decoder.decoders)):
                att_ws += [self.decoder.decoders[i].src_attn.attn]
            att_ws = torch.stack(att_ws, dim=1)  # (B, L, H, T_feats, T_text)

            return dict(feat_gen=outs[0], att_w=att_ws[0])

        # forward encoder
        xs = x.unsqueeze(0)
        if self.lang_family_encoding:
            hs, _ = self._multiple_encoding(xs, None, lids)
            # TODO(tsaeki): Output token embedding from multiple encoding
            token_emb = None
        elif self.use_encoder_w_lid:
            lid_embs = self.lid_emb(lids.view(-1))
            hs, _, token_emb, enc_in = self.encoder(xs, None, lid_embs, out_token_emb=True)
        else:
            hs, _, token_emb, enc_in = self.encoder(xs, None, out_token_emb=True)
        
        enc_out = hs.clone()

        # integrate GST
        if self.use_gst:
            style_embs = self.gst(y.unsqueeze(0))
            hs = hs + style_embs.unsqueeze(1)

        # integrate spk & lang embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            spembs = spemb.unsqueeze(0)
            hs = self._integrate_with_spk_embed(hs, spembs)

        # Integrate language embedding
        if self.lang_embed_dim is not None:
            lembs = lemb.unsqueeze(0)
            hs = self._integrate_with_lang_embed(hs, lembs)

        # Using speaker adapter
        if self.use_spk_adapter:
            hs = self.spk_adapter(hs)

        # set limits of length
        maxlen = int(hs.size(1) * maxlenratio / self.reduction_factor)
        minlen = int(hs.size(1) * minlenratio / self.reduction_factor)

        # initialize
        idx = 0
        ys = hs.new_zeros(1, 1, self.odim)
        outs, probs = [], []

        # forward decoder step-by-step
        z_cache = self.decoder.init_state(x)
        while True:
            # update index
            idx += 1

            # calculate output and stop prob at idx-th step
            y_masks = subsequent_mask(idx).unsqueeze(0).to(x.device)
            z, z_cache = self.decoder.forward_one_step(
                ys, y_masks, hs, cache=z_cache
            )  # (B, adim)
            outs += [
                self.feat_out(z).view(self.reduction_factor, self.odim)
            ]  # [(r, odim), ...]
            probs += [torch.sigmoid(self.prob_out(z))[0]]  # [(r), ...]

            # update next inputs
            ys = torch.cat(
                (ys, outs[-1][-1].view(1, 1, self.odim)), dim=1
            )  # (1, idx + 1, odim)

            # get attention weights
            att_ws_ = []
            for name, m in self.named_modules():
                if isinstance(m, MultiHeadedAttention) and "src" in name:
                    att_ws_ += [m.attn[0, :, -1].unsqueeze(1)]  # [(#heads, 1, T),...]
            if idx == 1:
                att_ws = att_ws_
            else:
                # [(#heads, l, T), ...]
                att_ws = [
                    torch.cat([att_w, att_w_], dim=1)
                    for att_w, att_w_ in zip(att_ws, att_ws_)
                ]

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = (
                    torch.cat(outs, dim=0).unsqueeze(0).transpose(1, 2)
                )  # (T_feats, odim) -> (1, T_feats, odim) -> (1, odim, T_feats)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, T_feats)
                outs = outs.transpose(2, 1).squeeze(0)  # (T_feats, odim)
                probs = torch.cat(probs, dim=0)
                break

        # concatenate attention weights -> (#layers, #heads, T_feats, T_text)
        att_ws = torch.stack(att_ws, dim=0)

        out_dict = dict(
            feat_gen=outs,
            prob=probs,
            att_w=att_ws,
            enc_out=enc_out
        )
        if token_emb is not None:
            out_dict.update(
                token_emb=token_emb,
                enc_in=enc_in
            )
        if (self.langs is not None) or self.use_encoder_w_lid:
            out_dict.update(lid_emb=lid_embs)
        return out_dict

    def _add_first_frame_and_remove_last_frame(self, ys: torch.Tensor) -> torch.Tensor:
        ys_in = torch.cat(
            [ys.new_zeros((ys.shape[0], 1, ys.shape[2])), ys[:, :-1]], dim=1
        )
        return ys_in

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

    def _target_mask(self, olens: torch.Tensor) -> torch.Tensor:
        """Make masks for masked self-attention.

        Args:
            olens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for masked self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> olens = [5, 3]
            >>> self._target_mask(olens)
            tensor([[[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]],
                    [[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        y_masks = make_non_pad_mask(olens).to(next(self.parameters()).device)
        s_masks = subsequent_mask(y_masks.size(-1), device=y_masks.device).unsqueeze(0)
        return y_masks.unsqueeze(-2) & s_masks

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _integrate_with_lang_embed(
        self, hs: torch.Tensor, lembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate language embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            lembs (Tensor): Batch of language embeddings (B, lang_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).
        """
        if self.lang_embed_integration_type == "add":
            # apply projection and then add to hidden states
            lembs = self.lang_projection(F.normalize(lembs))
            hs = hs + lembs.unsqueeze(1)
        elif self.lang_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            lembs = F.normalize(lembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.lang_projection(torch.cat([hs, lembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _bert_mask(self, xs, ilens):
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
        src_mask = make_non_pad_mask(ilens).to(next(self.parameters()).device)

        rand_mat = torch.rand(xs.shape).to(next(self.parameters()).device)
        ratio = ilens / xs.shape[1]
        rand_mat = rand_mat * ratio.unsqueeze(1)
        # Not masking <eos> token
        rand_mat[xs == self.eos] = 1.0

        rep_id = random.choice(range(self.mask+1, self.eos))
        mask_mask = rand_mat < 0.12
        mask_rep = (rand_mat > 0.12) & (rand_mat < 0.135)
        mask_pos = (rand_mat < 0.150) * src_mask
        xs_masked = xs.masked_fill_(mask_mask, self.mask)
        xs_masked = xs_masked.masked_fill_(mask_rep, rep_id)
        xs_masked = xs_masked * src_mask

        # mlm target (ignoring unmasked elements)
        mlm_target = xs.clone().masked_fill_(
            ~mask_pos, self.padding_idx
        )
        return xs_masked, mlm_target


class EncoderWithAdapter(Encoder):
    def forward(self, xs, masks, out_token_emb=False):
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

        if out_token_emb:
            token_emb = xs.clone()
        
        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                xs, masks = self.adapter(xs, masks)
            else:
                xs = self.adapter(xs)
        
        if out_token_emb:
            enc_in = xs.clone()

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
        if out_token_emb:
            return xs, masks, token_emb, enc_in
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
    def forward(self, xs, masks, lang_emb=None, out_token_emb=False):
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

        if out_token_emb:
            token_emb = xs.clone()
        
        # Adding language embedding
        if lang_emb is not None:
            xs = xs + lang_emb.unsqueeze(1)

        # Adding adapter
        if self.adapter is not None:
            if isinstance(self.adapter, TransformerAdapter):
                xs, masks = self.adapter(xs, masks)
            else:
                xs = self.adapter(xs)

        if out_token_emb:
            enc_in = xs.clone()

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
        if out_token_emb:
            return xs, masks, token_emb, enc_in
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


class LidPredictor(torch.nn.Module):
    """
    Language ID predictor module based on convolution
    """

    def __init__(self, in_dim, n_langs, level="utterance"):
        super().__init__()
        self.conv_block_1 = ConvBlockRes1D(
            in_channels=in_dim,
            out_channels=512,
            size=3)
        self.conv_block_2 = ConvBlockRes1D(
            in_channels=512,
            out_channels=1024,
            size=7)
        self.conv_block_3 = ConvBlockRes1D(
            in_channels=1024,
            out_channels=1024,
            size=7)
        self.conv_block_4 = ConvBlockRes1D(
            in_channels=1024,
            out_channels=512,
            size=3)
        self.linear_out = torch.nn.Linear(
            in_features=512,
            out_features=n_langs,
            bias=True)
        assert level in ("utterance", "token")
        self.level = level
        if level == "utterance":
            self.avgpool2d = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            output of analysis module: (batch, time, dim)
        Return:
            Prediction:
                if level == "utterance": (batch, n_langs)
                if level == "token": (batch, time, n_langs)
        """
        x = x.transpose(1, 2) # (batch, dim, time)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        if self.level == "utterance":
            x = self.avgpool2d(x)
            x = self.linear_out(x.squeeze(2))
        else:
            x = x.transpose(1, 2)
            x = self.linear_out(x)
        return x


class ConvBlockRes1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, size, momentum=0.01):
        super().__init__()

        pad = size // 2
        self.conv1 = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=1,
            dilation=1,
            padding=pad,
            bias=False)
        self.bn1 = torch.nn.BatchNorm1d(in_channels, momentum=momentum)
        self.conv2 = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=size,
            stride=1,
            dilation=1,
            padding=pad,
            bias=False)
        self.bn2 = torch.nn.BatchNorm1d(out_channels, momentum=momentum)
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0)
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


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

