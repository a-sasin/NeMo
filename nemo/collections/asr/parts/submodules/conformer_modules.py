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
#

from typing import Optional

import torch
from torch import nn as nn
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.adapters.attention_adapter_mixin import AttentionAdapterModuleMixin
from nemo.collections.asr.parts.submodules.batchnorm import FusedBatchNorm1d
from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import AccessMixin
import torch.nn.functional as F

__all__ = ['ConformerConvolution', 'ConformerFeedForward', 'ConformerLayer', 'ConformerMoEFeedForward', 'ConformerMoELayer', 'OmniRouter']


class ConformerLayer(torch.nn.Module, AttentionAdapterModuleMixin, AccessMixin):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'rel_pos_local_attn': relative positional embedding and Transformer-XL with local attention using
                overlapping chunks. Attention context is determined by att_context_size parameter.
            'abs_pos': absolute positional embedding and Transformer
            Default is rel_pos.
        global_tokens (int): number of tokens to be used for global attention.
            Only relevant if self_attention_model is 'rel_pos_local_attn'.
            Defaults to 0.
        global_tokens_spacing (int): how far apart the global tokens are
            Defaults to 1.
        global_attn_separate (bool): whether the q, k, v layers used for global tokens should be separate.
            Defaults to False.
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
        use_bias (bool): Apply bias to all Linear and Conv1d layers from each ConformerLayer to improve activation flow and stabilize training of huge models.
            Defaults to True.
    """

    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
    ):
        super(ConformerLayer, self).__init__()

        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5

        # first feed forward module
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )

        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]

        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )

        # second feed forward module
        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None

        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x

        residual = residual + self.dropout(x)

        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            residual = pack_input['x']

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(residual)

        if self.is_adapter_available():
            # Call the adapters
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = pack_input['x']

        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time


class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
        use_bias (bool): Use bias in all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
            Defaults to True
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        norm_type='batch_norm',
        conv_context_size=None,
        pointwise_activation='glu_',
        use_bias=True,
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.use_bias = use_bias

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=self.use_bias,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type == 'fused_batch_norm':
            self.batch_norm = FusedBatchNorm1d(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim,
            out_channels=d_model,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bias,
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model**-0.5
        dw_max = self.kernel_size**-0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            if self.use_bias:
                nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
                nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
                nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)


class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    use_bias (bool): Apply bias to all Linear and Conv1d layers improve activation flow and stabilize training of huge models.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish(), use_bias=True):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_bias = use_bias
        self.linear1 = nn.Linear(d_model, d_ff, bias=self.use_bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=self.use_bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)

class OmniRouter(nn.Module):
    """
    Omni Router for Mixture of Experts.
    Routes inputs to top-k experts based on learned routing weights.
    
    Args:
        d_model (int): input dimension
        num_experts (int): number of experts
        top_k (int): number of experts to route to
        use_bias (bool): whether to use bias in the router
        router_noise (float): standard deviation of noise added to router logits during training
    """
    
    def __init__(self, d_model, num_experts, top_k=2, use_bias=False, router_noise=0.1):
        super(OmniRouter, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_noise = router_noise
        
        # Router is a simple linear layer that outputs logits for each expert
        self.router = nn.Linear(d_model, num_experts, bias=use_bias)
        
    def forward(self, x, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, d_model)
            pad_mask (torch.Tensor, optional): bool tensor of shape (batch_size, seq_len)
                where True denotes padded frames.
        
        Returns:
            routing_weights (torch.Tensor): routing weights of shape (batch_size, seq_len, top_k)
            selected_experts (torch.Tensor): indices of selected experts of shape (batch_size, seq_len, top_k)
            router_logits (torch.Tensor): raw logits of shape (batch_size, seq_len, num_experts)
        """

        router_logits = self.router(x)  # (batch_size, seq_len, num_experts)
        
        # Add noise during training for load balancing
        if self.training and self.router_noise > 0:
            noise = torch.randn_like(router_logits) * self.router_noise
            router_logits_with_noise = router_logits + noise
        else:
            router_logits_with_noise = router_logits
        
        # Get top-k experts
        routing_weights, selected_experts = torch.topk(router_logits_with_noise, self.top_k, dim=-1)
        
        # Apply softmax to get normalized weights
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Return the logits that were actually used for routing (with noise during training)
        # This ensures load balance loss computes P_i from the same distribution used for selection
        return routing_weights, selected_experts, router_logits_with_noise

class ConformerMoEFeedForward(nn.Module):
    """
    Mixture of Experts feed-forward module for Conformer model.
    
    Args:
        d_model (int): input dimension
        d_ff (int): hidden dimension of each expert
        num_experts (int): number of experts
        top_k (int): number of experts to route to
        dropout (float): dropout probability
        activation: activation function
        use_bias (bool): whether to use bias in linear layers
        router_noise (float): standard deviation of noise added to router logits during training
    """
    
    def __init__(
        self, 
        d_model, 
        d_ff, 
        num_experts=8, 
        top_k=2, 
        dropout=0.1, 
        activation=Swish(), 
        use_bias=True,
        router_noise=0.1,
        router: Optional['OmniRouter'] = None,
        use_shared_router: bool = False,
    ):
        super(ConformerMoEFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_bias = use_bias
        self.dropout_rate = dropout
        self.use_shared_router = use_shared_router

        # Router selection policy:
        # - use_shared_router=True: consume the externally provided shared router.
        # - use_shared_router=False: ignore any external router and create a layer-local router.
        #
        # Shared router is stored via object.__setattr__ to avoid duplicate nn.Module
        # registration under each layer (the encoder owns registration and optimizer/state_dict).
        if self.use_shared_router:
            if router is None:
                raise ValueError("use_shared_router=True but no shared router was provided.")
            object.__setattr__(self, 'router', router)
        else:
            self.router = OmniRouter(d_model, num_experts, top_k, use_bias=False, router_noise=router_noise)
        
        # Create expert weights as stacked tensors for grouped matmul
        # Shape: (num_experts, d_ff, d_model) for w1 and (num_experts, d_model, d_ff) for w2
        self.w1 = nn.Parameter(torch.empty(num_experts, d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        
        if use_bias:
            self.b1 = nn.Parameter(torch.empty(num_experts, d_ff))
            self.b2 = nn.Parameter(torch.empty(num_experts, d_model))
        else:
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
        
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        
        # Flag to enable routing data capture during inference (for visualization)
        self.capture_routing_data = False
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize expert weights"""
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        
        with torch.no_grad():
            nn.init.uniform_(self.w1, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.w2, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.b1, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.b2, -ffn2_max, ffn2_max)
    
    def forward(self, x, pad_mask=None):
        """
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, d_model)
            pad_mask (torch.Tensor, optional): bool tensor of shape (batch_size, seq_len)
                where True denotes padded frames.
        
        Returns:
            output (torch.Tensor): output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get routing decisions - top k experts 
        routing_weights, selected_experts, router_logits = self.router(
            x, pad_mask=pad_mask
        )  # (B, T, top_k), (B, T, top_k), (B, T, num_experts)
        
        # Store for load balance loss computation (training) or visualization (inference)
        if self.training or self.capture_routing_data:
            # Keep gradient path for auxiliary load-balancing loss during training.
            # Detaching here prevents the router from receiving useful gradients.
            self._last_router_logits = router_logits
            self._last_routing_weights = routing_weights.detach()
            self._last_selected_experts = selected_experts.detach()
            self._last_pad_mask = pad_mask.detach() if pad_mask is not None else None
        
        # Flatten tokens for dispatch
        num_tokens = batch_size * seq_len
        x_flat = x.view(num_tokens, d_model)                               # (B*T, d_model)
        routing_weights_flat = routing_weights.view(num_tokens, self.top_k)   # (B*T, top_k)
        selected_experts_flat = selected_experts.view(num_tokens, self.top_k)  # (B*T, top_k)
        if pad_mask is not None:
            valid_mask_flat = (~pad_mask).view(num_tokens).to(x_flat.dtype)
        else:
            valid_mask_flat = x_flat.new_ones(num_tokens)

        # Accumulate expert contributions into output_flat.
        # Starting from zeros (no grad), each `output_flat + full_out * weights_k`
        # builds up a grad_fn chain that connects the output to BOTH:
        #
        #   (a) expert weights (w1, w2) via non-inplace .index_add()
        #       .index_add() returns a NEW tensor with grad_fn=IndexAddBackward so that
        #       gradients flow:  loss → full_out → expert_out → matmul → w1, w2
        #       (contrast with zeros[idx]=src or index_add_() on a leaf tensor, which
        #        silently severs the grad path because the leaf's requires_grad=False)
        #
        #   (b) routing weights (router) via  full_out * weights_k
        #       weights_k = routing_weights_flat[:, k] carries a grad_fn from softmax(topk)
        #       so loss → weighted → weights_k → softmax → router logits → W_router
        #
        # CRITICAL: ALL (k, expert_idx) iterations MUST execute on EVERY GPU regardless
        # of whether any tokens were assigned to that expert.  This keeps the DDP/NCCL
        # all-reduce stacks in sync and prevents distributed deadlocks.
        output_flat = x_flat.new_zeros(num_tokens, d_model)

        for k in range(self.top_k):
            # Exclude padded frames from routing contributions
            weights_k     = routing_weights_flat[:, k] * valid_mask_flat
            expert_ids_k  = selected_experts_flat[:, k]     # (B*T,)  int indices, no grad

            for expert_idx in range(self.num_experts):

                token_indices = (expert_ids_k == expert_idx).nonzero(as_tuple=True)[0]

                # Sparse expert FFN — torch.matmul handles shape-(0, d) tensors fine
                expert_input = x_flat[token_indices]                             # (n, d_model)
                hidden = torch.matmul(expert_input, self.w1[expert_idx].t())    # (n, d_ff)
                if self.use_bias:
                    hidden = hidden + self.b1[expert_idx]
                hidden = self.activation(hidden)
                hidden = self.dropout(hidden)
                expert_out = torch.matmul(hidden, self.w2[expert_idx].t())      # (n, d_model)
                if self.use_bias:
                    expert_out = expert_out + self.b2[expert_idx]

                # Non-inplace index_add: returns a NEW tensor with grad_fn=IndexAddBackward.
                # grad_expert_out[i] = grad_full_out[token_indices[i]]  (a simple gather)
                # This is the fix: expert weights now receive gradients from the task loss.
                full_out = x_flat.new_zeros(num_tokens, d_model).index_add(
                    0, token_indices, expert_out
                )                                                                # (B*T, d_model)

                # Scale by routing weight and accumulate — grad flows to both
                # full_out (→ w1, w2) and weights_k (→ router)
                output_flat = output_flat + full_out * weights_k.unsqueeze(-1)

        output = output_flat.view(batch_size, seq_len, d_model)
        if pad_mask is not None:
            output = output.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return output
    
    def reset_parameters_ff(self):
        """Reset parameters for all experts"""
        ffn1_max = self.d_model**-0.5
        ffn2_max = self.d_ff**-0.5
        
        with torch.no_grad():
            nn.init.uniform_(self.w1, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.w2, -ffn2_max, ffn2_max)
            if self.use_bias:
                nn.init.uniform_(self.b1, -ffn1_max, ffn1_max)
                nn.init.uniform_(self.b2, -ffn2_max, ffn2_max)
    
    def get_load_balance_loss(self):
        """
        Compute load balancing auxiliary loss as in Switch Transformer.
        
        Formula: L = num_experts * Σ(f_i * P_i)
        where:
            f_i = fraction of tokens routed to expert i (after top-k selection)
            P_i = mean routing probability for expert i (before top-k selection)
        
        This encourages the router to distribute load evenly across experts.
        
        Returns:
            torch.Tensor: scalar load balance loss
        """
        if not self.training or not hasattr(self, '_last_router_logits'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        router_logits = self._last_router_logits  # (B, T, num_experts)
        selected_experts = self._last_selected_experts  # (B, T, top_k)
        
        batch_size, seq_len, _ = router_logits.shape
        if hasattr(self, '_last_pad_mask') and self._last_pad_mask is not None:
            valid_mask = (~self._last_pad_mask).to(router_logits.dtype)  # (B, T)
        else:
            valid_mask = torch.ones(batch_size, seq_len, device=router_logits.device, dtype=router_logits.dtype)

        num_valid_tokens = valid_mask.sum().clamp_min(1.0)
        
        # Compute P_i: mean routing probability per expert (across all tokens)
        router_probs = F.softmax(router_logits, dim=-1)  # (B, T, num_experts)
        P_i = (router_probs * valid_mask.unsqueeze(-1)).sum(dim=[0, 1]) / num_valid_tokens  # (num_experts,)
        
        # Compute f_i: fraction of tokens actually routed to each expert
        # Create one-hot encoding of selected experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)  # (B, T, top_k, num_experts)
        expert_mask = expert_mask.sum(dim=2).float()  # (B, T, num_experts)
        expert_mask = expert_mask * valid_mask.unsqueeze(-1)
        f_i = expert_mask.sum(dim=[0, 1]) / (num_valid_tokens * self.top_k)  # (num_experts,)
        
        # Switch Transformer load balance loss: N * Σ(f_i * P_i)
        load_balance_loss = self.num_experts * torch.sum(f_i * P_i)
        
        return load_balance_loss


class ConformerMoELayer(torch.nn.Module, AttentionAdapterModuleMixin, AccessMixin):
    """
    A Conformer encoder layer with a single Mixture of Experts feed-forward module.

    The block order is: Self-Attention → Convolution → MoE Feed-Forward → Output.
    Unlike the standard ConformerLayer (which has two half-scaled FF modules sandwiching
    attention and convolution), this layer uses one MoE FF module placed after convolution.
    """
    
    def __init__(
        self,
        d_model,
        d_ff,
        self_attention_model='rel_pos',
        global_tokens=0,
        global_tokens_spacing=1,
        global_attn_separate=False,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_norm',
        conv_context_size=None,
        dropout=0.1,
        dropout_att=0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        use_bias=True,
        use_pytorch_sdpa=False,
        use_pytorch_sdpa_backends=None,
        # MoE-specific parameters
        num_experts=8,
        top_k=2,
        router_noise=0.1,
        load_balance_loss_weight=0.0,
        shared_router: Optional['OmniRouter'] = None,
        use_shared_router: bool = False,
    ):
        super(ConformerMoELayer, self).__init__()
        
        self.use_pytorch_sdpa = use_pytorch_sdpa
        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []
        self.use_pytorch_sdpa_backends = use_pytorch_sdpa_backends
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.fc_factor = 0.5
        self.use_shared_router = use_shared_router

        # store load balance loss weight
        self.load_balance_loss_weight = load_balance_loss_weight
        
        # convolution module
        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
        )
        
        # multi-headed self-attention module
        self.norm_self_att = LayerNorm(d_model)
        MHA_max_cache_len = att_context_size[0]
        
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=MHA_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=MHA_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=self.use_pytorch_sdpa,
                use_pytorch_sdpa_backends=self.use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not not a valid value for 'self_attention_model', "
                f"valid values can be from ['rel_pos', 'rel_pos_local_attn', 'abs_pos']"
            )
        
        # feed forward module (MoE)
        self.norm_feed_forward = LayerNorm(d_model)
        self.feed_forward = ConformerMoEFeedForward(
            d_model=d_model, 
            d_ff=d_ff, 
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout, 
            use_bias=use_bias,
            router_noise=router_noise,
            router=shared_router,
            use_shared_router=use_shared_router,
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)
    
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
            cache_last_channel (torch.tensor) : cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : cache for convolutional layers (B, d_model, T_cache)
        Returns:
            x (torch.Tensor): (B, T, d_model)
            cache_last_channel (torch.tensor) : next cache for MHA layers (B, T_cache, d_model)
            cache_last_time (torch.tensor) : next cache for convolutional layers (B, d_model, T_cache)
        """
        residual = x
        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'rel_pos_local_attn':
            x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, cache=cache_last_channel)
        else:
            x = None
        
        if x is not None and cache_last_channel is not None:
            (x, cache_last_channel) = x
        
        residual = residual + self.dropout(x)
        
        if self.is_adapter_available():
            # Call the MHA adapters
            pack_input = {
                'x': residual,
                'loc': 'mha',
                'att_mask': att_mask,
                'pos_emb': pos_emb,
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            residual = pack_input['x']
        
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            (x, cache_last_time) = x
        residual = residual + self.dropout(x)
        
        x = self.norm_feed_forward(residual)
        x = self.feed_forward(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x) * self.fc_factor
        
        x = self.norm_out(residual)
        
        if self.is_adapter_available():
            # Call the adapters
            pack_input = {
                'x': x,
                'loc': 'post',
            }
            pack_input = self.forward_enabled_adapters(pack_input)
            x = pack_input['x']
        
        if self.is_access_enabled(getattr(self, "model_guid", None)) and self.access_cfg.get(
            'save_encoder_tensors', False
        ):
            self.register_accessible_tensor(name='encoder', tensor=x)
        if cache_last_channel is None:
            return x
        else:
            return x, cache_last_channel, cache_last_time
    
    def get_load_balance_loss(self):
        """
        Collect load balance loss from both MoE feed-forward modules.
        
        Returns:
            torch.Tensor: weighted load balance loss from both FFN modules
        """
        loss = 0.0
        
        # Get loss from the MoE feed-forward module
        if hasattr(self.feed_forward, 'get_load_balance_loss'):
            loss += self.feed_forward.get_load_balance_loss()
        
        # Apply layer-specific weight
        return loss * self.load_balance_loss_weight
