# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ConformerMoEEncoder - A Conformer encoder with Mixture of Experts using Omni Router.

This is a separate architecture from the standard ConformerEncoder, specifically designed
for MoE training with the Omni Router pattern from Apple's research.

Key features:
- All layers use MoE feed-forward modules
- Shared Omni Router within each layer (across both FFN modules)
- Automatic load balancing loss collection
- Compatible with NeMo training infrastructure
"""

import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed
from omegaconf import DictConfig, ListConfig, open_dict
from torch import nn

from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerMoELayer, OmniRouter
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    LocalAttRelPositionalEncoding,
    MultiHeadAttention,
    PositionalEncoding,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.submodules.subsampling import (
    ConvSubsampling,
    StackingSubsampling,
    SubsamplingReductionModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.asr.parts.utils.regularization_utils import compute_stochastic_depth_drop_probs
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    BoolType,
    ChannelType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['ConformerMoEEncoder']


class ConformerMoEEncoder(NeuralModule, StreamingEncoder, Exportable, AccessMixin):
    """
    Conformer encoder with Mixture of Experts using Omni Router pattern.
    
    This is a dedicated MoE architecture where all layers use MoE feed-forward modules.
    Each layer has a shared Omni Router that routes to multiple expert feed-forward networks.
    
    Based on:
    - 'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
      https://arxiv.org/abs/2005.08100
    - 'Omni-Router: A Scalable and Efficient Router for Mixture of Experts' by Apple ML Research
      https://machinelearning.apple.com/research/omni-router
    
    Key Differences from Standard Conformer:
    - All feed-forward layers are MoE modules
    - Each layer has a shared Omni Router for both FFN modules
    - Automatic load balancing loss collection
    - Additional MoE-specific hyperparameters

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of MoE Conformer layers
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        
        # MoE-specific parameters
        num_experts (int): number of experts per MoE layer
            Defaults to 8.
        top_k (int): number of experts to route to per token
            Defaults to 2.
        router_noise (float): standard deviation of noise added to router logits during training
            Defaults to 0.01.
        load_balance_loss_weight (float): weight for the load balancing auxiliary loss
            Defaults to 0.01.
        
        # Standard Conformer parameters (same as ConformerEncoder)
        subsampling (str): the method of subsampling
        subsampling_factor (int): the subsampling factor
        subsampling_conv_chunking_factor(int): force chunk inputs
        subsampling_conv_channels (int): convolution channels in subsampling
        reduction (str, Optional): reduction method
        reduction_position (int, Optional): layer index for reduction
        reduction_factor (int): reduction factor
        ff_expansion_factor (int): expansion factor in feed forward layers
        self_attention_model (str): attention type ('rel_pos', 'rel_pos_local_attn', 'abs_pos')
        pos_emb_max_len (int): max length of positional embeddings
        n_heads (int): number of attention heads
        att_context_size: attention context sizes
        att_context_probs: context size probabilities
        att_context_style (str): 'regular' or 'chunked_limited'
        xscaling (bool): scale inputs by sqrt(d_model)
        untie_biases (bool): untie bias weights between layers
        conv_kernel_size (int): convolution kernel size
        conv_norm_type (str): normalization type in conv modules
        conv_context_size: convolution context sizes
        use_bias (bool): use bias in Linear/Conv1d layers
        dropout (float): dropout rate
        dropout_pre_encoder (float): dropout before encoder
        dropout_emb (float): dropout for positional embeddings
        dropout_att (float): dropout for attention
        stochastic_depth_drop_prob (float): layer drop probability
        stochastic_depth_mode (str): 'linear' or 'uniform'
        stochastic_depth_start_layer (int): start layer for stochastic depth
        global_tokens (int): number of global attention tokens
        global_tokens_spacing (int): spacing between global tokens
        global_attn_separate (bool): separate q/k/v for global tokens
        use_pytorch_sdpa (bool): use PyTorch SDPA
        use_pytorch_sdpa_backends: SDPA backend names
        sync_max_audio_length (bool): sync max length across GPUs
    """

    def input_example(self, max_batch=1, max_dim=256):
        """Generates input examples for tracing etc."""
        dev = next(self.parameters()).device
        if self.export_cache_support:
            window_size = max_dim
            if self.streaming_cfg is not None:
                if isinstance(self.streaming_cfg.chunk_size, list):
                    chunk_size = self.streaming_cfg.chunk_size[1]
                else:
                    chunk_size = self.streaming_cfg.chunk_size
                if isinstance(self.streaming_cfg.pre_encode_cache_size, list):
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size[1]
                else:
                    pre_encode_cache_size = self.streaming_cfg.pre_encode_cache_size
                window_size = chunk_size + pre_encode_cache_size
            input_example = torch.randn(max_batch, self._feat_in, window_size, device=dev)
            input_example_length = torch.randint(
                window_size // 4, window_size, (max_batch,), device=dev, dtype=torch.int64
            )
            cache_last_channel, cache_last_time, cache_last_channel_len = self.get_initial_cache_state(
                batch_size=max_batch, device=dev, max_dim=max_dim
            )
            all_input_example = tuple(
                [
                    input_example,
                    input_example_length,
                    cache_last_channel.transpose(0, 1),
                    cache_last_time.transpose(0, 1),
                    cache_last_channel_len,
                ]
            )
        else:
            input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
            input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
            all_input_example = tuple([input_example, input_example_length])

        return all_input_example

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "cache_last_channel_next": NeuralType(('D', 'B', 'T', 'D'), ChannelType(), optional=True),
                "cache_last_time_next": NeuralType(('D', 'B', 'D', 'T'), ChannelType(), optional=True),
                "cache_last_channel_next_len": NeuralType(tuple('B'), LengthsType(), optional=True),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel", "cache_last_time", "cache_last_channel_len"])
        else:
            return set()

    @property
    def disabled_deployment_output_names(self):
        if not self.export_cache_support:
            return set(["cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"])
        else:
            return set()

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        # MoE-specific parameters
        num_experts=8,
        top_k=2,
        router_noise=0.01,
        load_balance_loss_weight=0.01,
        use_shared_router: bool = False,
        # Standard Conformer parameters
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
    ):
        super().__init__()
        
        # Store MoE-specific parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_noise = router_noise
        self.load_balance_loss_weight = load_balance_loss_weight
        self.use_shared_router = use_shared_router
        
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.n_layers = n_layers
        self._feat_in = feat_in
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = subsampling_conv_chunking_factor

        self.self_attention_model = self_attention_model
        self.global_tokens = global_tokens
        self.global_attn_separate = global_attn_separate
        self.global_tokens_spacing = global_tokens_spacing
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.sync_max_audio_length = sync_max_audio_length

        # Setting up the att_context_size
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            conv_context_size=conv_context_size,
            conv_kernel_size=conv_kernel_size,
        )

        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None

        # Subsampling
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling in ['stacking', 'stacking_norm']:
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    norm=True if subsampling == 'stacking_norm' else False,
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
                    activation=nn.ReLU(True),
                    is_causal=causal_downsampling,
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        # Reduction
        if reduction and reduction_factor > 1:
            assert reduction_position >= -1 and reduction_position < n_layers
            self.reduction_subsampling = SubsamplingReductionModule(
                reduction=reduction,
                d_model=d_model,
                reduction_factor=reduction_factor,
            )
            self.reduction_position = reduction_position
        else:
            self.reduction_subsampling = None
            self.reduction_position = None

        self._feat_out = d_model

        # Biases for relative positional encoding
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        # Positional encodings
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout_pre_encoder,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            if max(att_context_size) <= 0:
                raise ValueError("When using local attention, context size must be set > 0")
            self.pos_enc = LocalAttRelPositionalEncoding(
                att_context_size=att_context_size,
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout_pre_encoder, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")

        # Optional global shared router: one OmniRouter reused across all MoE layers.
        # If disabled, each layer creates its own local router.
        if self.use_shared_router:
            self.global_router = OmniRouter(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                use_bias=False,
                router_noise=router_noise,
            )
        else:
            self.global_router = None

        # Create MoE Conformer layers – all share the same global router
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerMoELayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                conv_context_size=self.conv_context_size,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                att_context_size=self.att_context_size,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
                # MoE-specific parameters
                num_experts=num_experts,
                top_k=top_k,
                router_noise=router_noise,
                load_balance_loss_weight=load_balance_loss_weight,
                shared_router=self.global_router,
                use_shared_router=self.use_shared_router,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
            
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True

        self.setup_streaming_params()
        self.export_cache_support = False

        self.layer_drop_probs = compute_stochastic_depth_drop_probs(
            len(self.layers), stochastic_depth_drop_prob, stochastic_depth_mode, stochastic_depth_start_layer
        )
        
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_capture_at_layers = None

        router_mode = "global shared router" if self.use_shared_router else f"{n_layers} local routers"
        
        logging.info(
            f"Created ConformerMoEEncoder with {n_layers} layers, "
            f"{num_experts} experts per layer, top-{top_k} routing [{router_mode}]"
        )

    def forward_for_export(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        """Forward function for model export."""
        if cache_last_channel is not None:
            cache_last_channel = cache_last_channel.transpose(0, 1)
            cache_last_time = cache_last_time.transpose(0, 1)

        rets = self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        rets = self.streaming_post_process(rets, keep_all_outputs=False)
        if len(rets) == 2:
            return rets
        elif rets[2] is None and rets[3] is None and rets[4] is None:
            return (rets[0], rets[1])
        else:
            return (
                rets[0],
                rets[1],
                rets[2].transpose(0, 1),
                rets[3].transpose(0, 1),
                rets[4],
            )

    def streaming_post_process(self, rets, keep_all_outputs=True):
        """Post-process the output of the forward function for streaming."""
        if len(rets) == 2:
            return rets[0], rets[1], None, None, None

        (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len) = rets

        if cache_last_channel_next is not None and self.streaming_cfg.last_channel_cache_size >= 0:
            if self.streaming_cfg.last_channel_cache_size > 0:
                cache_last_channel_next = cache_last_channel_next[
                    :, :, -self.streaming_cfg.last_channel_cache_size :, :
                ]

        if self.streaming_cfg.valid_out_len > 0 and (not keep_all_outputs or self.att_context_style == "regular"):
            encoded = encoded[:, :, : self.streaming_cfg.valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=self.streaming_cfg.valid_out_len)

        return (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len)

    @typecheck()
    def forward(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
    ):
        """
        Forward function for the ConformerMoEEncoder.
        
        Args:
            audio_signal: audio features (batch, feat_in, n_frames)
            length: length of each sequence
            cache_last_channel: cache for attention layers
            cache_last_time: cache for convolution layers
            cache_last_channel_len: cache lengths
            
        Returns:
            outputs: encoded representations
            encoded_lengths: lengths of encoded sequences
            [caches if provided]
        """
        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
    ):
        """Internal forward pass implementation."""
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        # Select attention context size
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        # Pre-encoding
        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
            length = length.to(torch.int64)
            if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
                audio_signal = audio_signal[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
                length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        if self.reduction_position is not None and cache_last_channel is not None:
            raise ValueError("Caching with reduction feature is not supported yet!")

        max_audio_length = audio_signal.size(1)
        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
            padding_length = length + cache_len
            offset = torch.neg(cache_last_channel_len) + cache_len
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None

        audio_signal, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)

        # Create masks
        pad_mask, att_mask = self._create_masks(
            att_context_size=cur_att_context_size,
            padding_length=padding_length,
            max_audio_length=max_audio_length,
            offset=offset,
            device=audio_signal.device,
        )

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
            if att_mask is not None:
                att_mask = att_mask[:, cache_len:]
            cache_last_time_next = []
            cache_last_channel_next = []

        # Forward through MoE Conformer layers
        for lth, (drop_prob, layer) in enumerate(zip(self.layer_drop_probs, self.layers)):
            original_signal = audio_signal
            if cache_last_channel is not None:
                cache_last_channel_cur = cache_last_channel[lth]
                cache_last_time_cur = cache_last_time[lth]
            else:
                cache_last_channel_cur = None
                cache_last_time_cur = None
                
            audio_signal = layer(
                x=audio_signal,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=cache_last_channel_cur,
                cache_last_time=cache_last_time_cur,
            )

            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, cache_last_time_cur) = audio_signal
                cache_last_channel_next.append(cache_last_channel_cur)
                cache_last_time_next.append(cache_last_time_cur)

            # Stochastic depth
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1) < drop_prob
                if should_drop:
                    audio_signal = audio_signal * 0.0 + original_signal
                else:
                    audio_signal = (audio_signal - original_signal) / (1.0 - drop_prob) + original_signal

            # Reduction
            if self.reduction_position == lth:
                audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)
                max_audio_length = audio_signal.size(1)
                _, pos_emb = self.pos_enc(x=audio_signal, cache_len=cache_len)
                pad_mask, att_mask = self._create_masks(
                    att_context_size=cur_att_context_size,
                    padding_length=length,
                    max_audio_length=max_audio_length,
                    offset=offset,
                    device=audio_signal.device,
                )

            # InterCTC capture
            if self.is_access_enabled(getattr(self, "model_guid", None)):
                if self.interctc_capture_at_layers is None:
                    self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
                if lth in self.interctc_capture_at_layers:
                    lth_audio_signal = audio_signal
                    if self.out_proj is not None:
                        lth_audio_signal = self.out_proj(audio_signal)
                    self.register_accessible_tensor(
                        name=f'interctc/layer_output_{lth}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                    )
                    self.register_accessible_tensor(name=f'interctc/layer_length_{lth}', tensor=length)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        # Final reduction
        if self.reduction_position == -1:
            audio_signal, length = self.reduction_subsampling(x=audio_signal, lengths=length)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        length = length.to(dtype=torch.int64)

        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)
            cache_last_time_next = torch.stack(cache_last_time_next, dim=0)
            return (
                audio_signal,
                length,
                cache_last_channel_next,
                cache_last_time_next,
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            return audio_signal, length

    def get_load_balance_loss(self):
        """
        Collect load balancing losses from all MoE layers.
        
        Returns:
            total_lb_loss: sum of load balancing losses from all layers
        """
        total_lb_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, ConformerMoELayer):
                total_lb_loss += layer.get_load_balance_loss()
        return total_lb_loss

    def update_max_seq_length(self, seq_length: int, device):
        """Updates the maximum sequence length for the model."""
        if self.sync_max_audio_length and torch.distributed.is_initialized():
            global_max_len = torch.tensor([seq_length], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(global_max_len, op=torch.distributed.ReduceOp.MAX)
            seq_length = global_max_len.int().item()

        if seq_length > self.max_audio_length:
            self.set_max_audio_length(seq_length)

    def set_max_audio_length(self, max_audio_length):
        """Sets maximum input length and pre-calculates internal seq_range mask."""
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        self.pos_enc.extend_pe(max_audio_length, device, dtype)

    def _create_masks(self, att_context_size, padding_length, max_audio_length, offset, device):
        """Create attention and padding masks."""
        if self.self_attention_model != "rel_pos_local_attn":
            att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)

            if self.att_context_style == "regular":
                if att_context_size[0] >= 0:
                    att_mask = att_mask.triu(diagonal=-att_context_size[0])
                if att_context_size[1] >= 0:
                    att_mask = att_mask.tril(diagonal=att_context_size[1])
            elif self.att_context_style == "chunked_limited":
                if att_context_size[1] == -1:
                    if att_context_size[0] >= 0:
                        att_mask = att_mask.triu(diagonal=-att_context_size[0])
                else:
                    chunk_size = att_context_size[1] + 1
                    if att_context_size[0] >= 0:
                        left_chunks_num = att_context_size[0] // chunk_size
                    else:
                        left_chunks_num = 10000

                    chunk_idx = torch.arange(0, max_audio_length, dtype=torch.int, device=att_mask.device)
                    chunk_idx = torch.div(chunk_idx, chunk_size, rounding_mode="trunc")
                    diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                    chunked_limited_mask = torch.logical_and(
                        torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
                    )
                    att_mask = torch.logical_and(att_mask, chunked_limited_mask.unsqueeze(0))
        else:
            att_mask = None

        # Padding mask
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if offset is not None:
            pad_mask_off = torch.arange(0, max_audio_length, device=device).expand(
                padding_length.size(0), -1
            ) >= offset.unsqueeze(-1)
            pad_mask = pad_mask_off.logical_and(pad_mask)

        if att_mask is not None:
            pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
            pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
            att_mask = att_mask[:, :max_audio_length, :max_audio_length]
            att_mask = torch.logical_and(pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device))
            att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def enable_pad_mask(self, on=True):
        """Enables or disables the pad mask."""
        mask = self.use_pad_mask
        self.use_pad_mask = on
        return mask

    def _calc_context_sizes(
        self, att_context_size, att_context_probs, att_context_style, conv_context_size, conv_kernel_size
    ):
        """Calculate and validate context sizes."""
        # Convert att_context_size to standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if isinstance(att_cs, ListConfig):
                    att_context_size_all[i] = list(att_cs)
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(f"att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!")
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context (att_context_size[{i}][1]) cannot be unlimited for chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError("The sum of att_context_probs should equal one.")
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)

        if conv_context_size is not None:
            if isinstance(conv_context_size, ListConfig):
                conv_context_size = list(conv_context_size)
            if not isinstance(conv_context_size, list) and not isinstance(conv_context_size, str):
                raise ValueError("Invalid conv_context_size! Must be 'causal' or a list of two integers.")
            if conv_context_size == "causal":
                conv_context_size = [conv_kernel_size - 1, 0]
            else:
                if conv_context_size[0] + conv_context_size[1] + 1 != conv_kernel_size:
                    raise ValueError(f"Invalid conv_context_size: {conv_context_size}!")
        else:
            conv_context_size = [(conv_kernel_size - 1) // 2, (conv_kernel_size - 1) // 2]
            
        return att_context_size_all, att_context_size_all[0], att_context_probs, conv_context_size

    def set_default_att_context_size(self, att_context_size):
        """Sets the default attention context size."""
        if att_context_size not in self.att_context_size_all:
            logging.warning(
                f"att_context_size={att_context_size} is not among the supported look-aheads: {self.att_context_size_all}"
            )
        if att_context_size is not None:
            self.att_context_size = att_context_size
        self.setup_streaming_params()

    def setup_streaming_params(
        self,
        chunk_size: int = None,
        shift_size: int = None,
        left_chunks: int = None,
        att_context_size: list = None,
        max_context: int = 10000,
    ):
        """Setup streaming parameters."""
        streaming_cfg = CacheAwareStreamingConfig()

        if att_context_size is None:
            att_context_size = self.att_context_size

        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size must be >= 1")
            lookahead_steps = chunk_size - 1
            streaming_cfg.cache_drop_size = chunk_size - shift_size
        elif self.att_context_style == "chunked_limited":
            lookahead_steps = att_context_size[1]
            streaming_cfg.cache_drop_size = 0
        elif self.att_context_style == "regular":
            lookahead_steps = att_context_size[1] * self.n_layers + self.conv_context_size[1] * self.n_layers
            streaming_cfg.cache_drop_size = lookahead_steps
        else:
            streaming_cfg.cache_drop_size = 0
            lookahead_steps = None

        if chunk_size is None:
            streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
        else:
            if left_chunks is None:
                streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
                logging.warning(f"left_chunks not set. Using default: {streaming_cfg.last_channel_cache_size}")
            else:
                streaming_cfg.last_channel_cache_size = left_chunks * chunk_size

        if hasattr(self.pre_encode, "get_sampling_frames"):
            sampling_frames = self.pre_encode.get_sampling_frames()
        else:
            sampling_frames = 0

        if isinstance(sampling_frames, list):
            streaming_cfg.chunk_size = [
                sampling_frames[0] + self.subsampling_factor * lookahead_steps,
                sampling_frames[1] + self.subsampling_factor * lookahead_steps,
            ]
        else:
            streaming_cfg.chunk_size = sampling_frames * (1 + lookahead_steps)

        if isinstance(sampling_frames, list):
            streaming_cfg.shift_size = [
                sampling_frames[0] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
                sampling_frames[1] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
            ]
        else:
            streaming_cfg.shift_size = sampling_frames * (1 + lookahead_steps - streaming_cfg.cache_drop_size)

        if isinstance(streaming_cfg.shift_size, list):
            streaming_cfg.valid_out_len = (streaming_cfg.shift_size[1] - sampling_frames[1]) // self.subsampling_factor + 1
        else:
            streaming_cfg.valid_out_len = streaming_cfg.shift_size // self.subsampling_factor

        if hasattr(self.pre_encode, "get_streaming_cache_size"):
            streaming_cfg.pre_encode_cache_size = self.pre_encode.get_streaming_cache_size()
        else:
            streaming_cfg.pre_encode_cache_size = 0

        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            if streaming_cfg.pre_encode_cache_size[1] >= 1:
                streaming_cfg.drop_extra_pre_encoded = (1 + (streaming_cfg.pre_encode_cache_size[1] - 1) // self.subsampling_factor)
            else:
                streaming_cfg.drop_extra_pre_encoded = 0
        else:
            streaming_cfg.drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor

        for m in self.layers.modules():
            if hasattr(m, "_max_cache_len"):
                if isinstance(m, MultiHeadAttention):
                    m.cache_drop_size = streaming_cfg.cache_drop_size
                if isinstance(m, CausalConv1D):
                    m.cache_drop_size = streaming_cfg.cache_drop_size

        self.streaming_cfg = streaming_cfg

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):
        """Get initial cache states for streaming."""
        if device is None:
            device = next(self.parameters()).device
        if max_dim > 0:
            create_tensor = torch.randn
        else:
            create_tensor = torch.zeros
            
        last_time_cache_size = self.conv_context_size[0]
        cache_last_channel = create_tensor(
            (len(self.layers), batch_size, self.streaming_cfg.last_channel_cache_size, self.d_model),
            device=device,
            dtype=dtype,
        )
        cache_last_time = create_tensor(
            (len(self.layers), batch_size, self.d_model, last_time_cache_size),
            device=device,
            dtype=dtype,
        )
        
        if max_dim > 0:
            cache_last_channel_len = torch.randint(
                0,
                min(max_dim, self.streaming_cfg.last_channel_cache_size),
                (batch_size,),
                device=device,
                dtype=torch.int64,
            )
            for i in range(batch_size):
                cache_last_channel[:, i, cache_last_channel_len[i] :, :] = 0
                if cache_last_channel_len[i] == 0:
                    cache_last_time[:, i, :, :] = 0
        else:
            cache_last_channel_len = torch.zeros(batch_size, device=device, dtype=torch.int64)
            
        return cache_last_channel, cache_last_time, cache_last_channel_len


# Register adapter support
class ConformerMoEEncoderAdapter(ConformerMoEEncoder, adapter_mixins.AdapterModuleMixin):
    """ConformerMoEEncoder with adapter support."""

    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for conformer_layer in self.layers:
            conformer_layer.add_adapter(name, cfg)

    def is_adapter_available(self) -> bool:
        return any([conformer_layer.is_adapter_available() for conformer_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for conformer_layer in self.layers:
            conformer_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for conformer_layer in self.layers:
            names.update(conformer_layer.get_enabled_adapters())
        return sorted(list(names))

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg

    def get_accepted_adapter_types(self) -> Set[type]:
        types = super().get_accepted_adapter_types()
        if len(types) == 0:
            self.set_accepted_adapter_types(
                [
                    adapter_utils.LINEAR_ADAPTER_CLASSPATH,
                    adapter_utils.MHA_ADAPTER_CLASSPATH,
                    adapter_utils.RELMHA_ADAPTER_CLASSPATH,
                ]
            )
            types = self.get_accepted_adapter_types()
        return types


# Register adapter
if adapter_mixins.get_registered_adapter(ConformerMoEEncoder) is None:
    adapter_mixins.register_adapter(base_class=ConformerMoEEncoder, adapter_class=ConformerMoEEncoderAdapter)


@dataclass
class ConformerMoEConfig:
    """Configuration for ConformerMoEEncoder."""
    
    # Model dimensions
    feat_in: int = 80
    n_layers: int = 18
    d_model: int = 512
    feat_out: int = -1
    
    # MoE-specific
    num_experts: int = 8
    top_k: int = 2
    router_noise: float = 0.01
    load_balance_loss_weight: float = 0.01
    use_shared_router: bool = False
    
    # Standard Conformer parameters
    ff_expansion_factor: int = 4
    n_heads: int = 8
    conv_kernel_size: int = 31
    dropout: float = 0.1
    dropout_att: float = 0.0
