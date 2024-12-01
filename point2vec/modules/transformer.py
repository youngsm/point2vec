from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
# from timm.models.layers import DropPath
from torch import nn
import torch.nn.functional as F
from point2vec.modules.masking import MaskedLayerNorm, MaskedDropPath
import math

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,

    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, use_flash_attn=True):
        B, N, C = x.shape

        # Split into q, k, v and reshape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ) # (3, B, H, N, dim-per-head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if use_flash_attn:
            # Prepare attn_mask
            if attn_mask is not None:
                # (B, 1, N, N) --> (B * num_heads, N, N)
                attn_mask = attn_mask.squeeze(1).repeat_interleave(self.num_heads, dim=0)

            # Reshape q, k, v for scaled_dot_product_attention
            q = q.reshape(B * self.num_heads, N, self.head_dim)
            k = k.reshape(B * self.num_heads, N, self.head_dim)
            v = v.reshape(B * self.num_heads, N, self.head_dim)

            # Perform scaled dot product attention using PyTorch's built-in function
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p
            )

            # Reshape and project the output
            attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim)
            x = attn_output.transpose(1, 2).reshape(B, N, C)

            _attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if attn_mask is not None:
                attn = attn + attn_mask  # attn_mask should be broadcastable to attn

            attn = attn.softmax(dim=-1)
            _attn = attn
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, _attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=MaskedLayerNorm,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = MaskedDropPath(drop_path) if drop_path > 0.0 else Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # MLP BLOCK
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, attn_mask=None, embedding_mask=None, use_flash_attn=True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _x, attn = self.attn(self.norm1(x, embedding_mask), attn_mask, use_flash_attn)
        x = x + self.drop_path(_x, embedding_mask)
        ffn = self.mlp(self.norm2(x, embedding_mask))
        if embedding_mask is not None:
            ffn = ffn * embedding_mask.unsqueeze(-1)
        x = x + self.drop_path(ffn, embedding_mask)
        return x, attn, ffn

@dataclass()
class TransformerEncoderOutput:
    last_hidden_state: torch.Tensor  # (B, T, C)
    hidden_states: Optional[List[torch.Tensor]] = None  # [(B, T, C)]
    attentions: Optional[List[torch.Tensor]] = None  # [(B, H, T)]
    ffns: Optional[List[torch.Tensor]] = None  # [(B, T, C)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate: float | List[float] = 0.0,
        add_pos_at_every_layer=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i]
                    if isinstance(drop_path_rate, list)
                    else drop_path_rate,
                )
                for i in range(depth)
            ]
        )

        # output norm
        self.norm = MaskedLayerNorm(embed_dim)

        self.add_pos_at_every_layer = add_pos_at_every_layer

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        embedding_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ) -> TransformerEncoderOutput:

        if embedding_mask is not None:
            B, N, C = x.shape
            assert embedding_mask.shape == (
                B,
                N,
            ), "embedding_mask must be of shape (B, N)"

            # Create additive attention mask of shape (B, 1, N, N)
            attn_mask = (
                embedding_mask.unsqueeze(1).unsqueeze(2) & embedding_mask.unsqueeze(1).unsqueeze(3)
            )
            attn_mask = (
                (~attn_mask).to(x.dtype)
                .masked_fill(~attn_mask, -1e9)
            )

        else:
            attn_mask = None

        hidden_states = [] if return_hidden_states else None
        attentions = [] if return_attentions else None
        ffns = [] if return_ffns else None
        if not self.add_pos_at_every_layer:
            x = x + pos
        for block in self.blocks:
            if self.add_pos_at_every_layer:
                x = x + pos
            x, attn, ffn = block(x, attn_mask, embedding_mask, use_flash_attn=not return_attentions)
            if return_hidden_states:
                assert hidden_states is not None
                hidden_states.append(x)
            if return_attentions:
                assert attentions is not None
                attentions.append(attn)
            if return_ffns:
                assert ffns is not None
                ffns.append(ffn)
        x = self.norm(x, embedding_mask)
        return TransformerEncoderOutput(x, hidden_states, attentions, ffns)


class PromptedTransformerEncoder(TransformerEncoder):
    def __init__(
            self,
            embed_dim=768,
            depth=4,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate: float | List[float] = 0.0,
            add_pos_at_every_layer=False,
            num_channels=4,
            points_per_token=128,
            num_prompts=10,
            prompt_dropout=0.0,
            deep=False,
    ):
        super().__init__(embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, add_pos_at_every_layer)

        self.num_prompts = num_prompts
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_prompts, embed_dim))
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.deep = deep

        fan_in = num_channels * points_per_token
        fan_out = embed_dim
        val = math.sqrt(6. / (fan_in + fan_out))
        nn.init.uniform_(self.prompt_embeddings, -val, val)

        # xavier init prompt embeddings
        if deep:
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(len(self.blocks)-1, num_prompts, embed_dim))
            nn.init.uniform_(self.deep_prompt_embeddings, -val, val)

    def integrate_prompt(
            self,
            x: torch.Tensor,
            prompts: torch.Tensor,
            pos: torch.Tensor | None = None,
            embedding_mask: torch.Tensor | None = None
    ):
        """
        x: (B, N, C)
        prompts: (1, Np, C)
        pos: (B, N, C)
        embedding_mask: (B, N)
        """
        if prompts.dim() == 2:
            prompts = prompts.unsqueeze(0)

        B, N, C = x.shape
        _, Np, _ = prompts.shape
        device = x.device

        prompts = self.prompt_dropout(prompts.expand(B, -1, -1))
        x = torch.cat([prompts, x], dim=1)

        if pos is not None: # (B,N,C) --> (B,Np+N, C)
            pos = torch.cat([torch.zeros(B, Np, C, device=device, dtype=pos.dtype), pos], dim=1)

        if embedding_mask is not None:
            # add ones in front of embedding mask for each batch
            embedding_mask = torch.cat([torch.ones(B, Np, device=device, dtype=embedding_mask.dtype), embedding_mask], dim=1)

        return x, pos, embedding_mask
    
    def reintegrate_prompt(
            self,
            x: torch.Tensor,
            prompts: torch.Tensor,
            pos: torch.Tensor | None = None,
            embedding_mask: torch.Tensor | None = None
    ):
        x = x[:, self.num_prompts:, :]

        if pos is not None:
            pos = pos[:, self.num_prompts:, :]

        if embedding_mask is not None:
            embedding_mask = embedding_mask[:, self.num_prompts:]

        x, pos, embedding_mask = self.integrate_prompt(x, prompts, pos, embedding_mask)
        return x, pos, embedding_mask

    def train(self, mode=True):
        if mode:
            self.blocks.eval()
            self.prompt_dropout.train()
            self.norm.eval()
        else:
            for module in self.children():
                module.train(mode)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        embedding_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_ffns: bool = False,
    ):
        x, pos, embedding_mask = self.integrate_prompt(x, self.prompt_embeddings, pos, embedding_mask)

        if embedding_mask is not None:
            B, N, C = x.shape
            assert embedding_mask.shape == (
                B,
                N,
            ), "embedding_mask must be of shape (B, N)"

            # Create additive attention mask of shape (B, 1, N, N)
            attn_mask = (
                embedding_mask.unsqueeze(1).unsqueeze(2) & embedding_mask.unsqueeze(1).unsqueeze(3)
            )
            attn_mask = (
                (~attn_mask).to(x.dtype)
                .masked_fill(~attn_mask, -1e9)
            )

        else:
            attn_mask = None

        hidden_states = [] if return_hidden_states else None
        attentions = [] if return_attentions else None
        ffns = [] if return_ffns else None
        if not self.add_pos_at_every_layer:
            x = x + pos
        for i, block in enumerate(self.blocks):
            if self.add_pos_at_every_layer:
                x = x + pos

            if i != 0 and self.deep:
                x = self.reintegrate_prompt(x, self.deep_prompt_embeddings[i-1])[0]
            x, attn, ffn = block(x, attn_mask, embedding_mask, use_flash_attn=not return_attentions)

            if return_hidden_states:
                assert hidden_states is not None
                hidden_states.append(x)
            if return_attentions:
                assert attentions is not None
                attentions.append(attn)
            if return_ffns:
                assert ffns is not None
                ffns.append(ffn)
        x = self.norm(x, embedding_mask)
        return TransformerEncoderOutput(x, hidden_states, attentions, ffns)