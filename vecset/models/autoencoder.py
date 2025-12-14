# Copyright (c) 2025, Biao Zhang.

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch_cluster import fps

from .bottleneck import Bottleneck, KLBottleneck, NormalizedBottleneck
from .utils import Attention, FeedForward, PointEmbed, PreNorm, subsample


class VecSetAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=768,
        output_dim=1,
        num_inputs=2048,
        num_latents=1280,
        latent_dim=16,
        dim_head=64,
        query_type="point",
        bottleneck=None,
        bottleneck_args={},
    ):
        super().__init__()

        queries_dim = dim

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents  # Latent VecSet의 Size. 각 vector는 dim 차원

        self.query_type = query_type
        if query_type == "point":
            pass
        elif query_type == "learnable":
            self.latents = nn.Embedding(num_latents, dim)
        else:
            raise NotImplementedError(f"Query type {query_type} not implemented")

        # 인코더 앞단: Attention + MLP (FeedForward)가 합쳐진 구조
        self.cross_attend_blocks = nn.ModuleList([PreNorm(dim, Attention(dim, dim, heads=dim // dim_head, dim_head=dim_head)), PreNorm(dim, FeedForward(dim))])

        # Fourier Feature Embedding + 합쳐진 정보를 Linear Layer에 통과시켜, 모델이 처리할 수 있는 크기(dim)로 변환하고 특징을 섞어줌
        self.point_embed = PointEmbed(dim=dim)

        # 인코더 본체: 여러 층의 Self-Attention + MLP (FeedForward)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([PreNorm(dim, Attention(dim, heads=dim // dim_head, dim_head=dim_head)), PreNorm(dim, FeedForward(dim))]))

        # 디코더 (Simple Attention. MLP 없음)
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads=dim // dim_head, dim_head=dim_head))

        # Output Head. Linear Projection
        self.to_outputs = nn.Sequential(
            nn.LayerNorm(queries_dim),
            nn.Linear(queries_dim, output_dim),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.to_outputs[1].weight)
        nn.init.zeros_(self.to_outputs[1].bias)

        self.bottleneck = bottleneck(**bottleneck_args)

    # 입력된 Point Cloud(pc)에서 핵심 정보를 추출하여 Latent Code로 변환하는 역할
    def encode(self, pc):
        B, N, _ = pc.shape
        assert N == self.num_inputs

        if self.query_type == "point":
            sampled_pc = subsample(pc, N, self.num_latents)
            x = self.point_embed(sampled_pc)
        elif self.query_type == "learnable":
            x = repeat(self.latents.weight, "n d -> b n d", b=B)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(x, context=pc_embeddings, mask=None) + x
        x = cross_ff(x) + x

        bottleneck = self.bottleneck.pre(x)
        return bottleneck

    # 전역적인 구조와 특징이 서로 융합되고, 불필요한 노이즈가 제거된 깨끗한 최종 Latent Code를 생성
    def learn(self, x):
        x = self.bottleneck.post(x)

        if self.query_type == "learnable":
            x = x + self.latents.weight[None]

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x

    # Latent Code(x)와 우리가 알고 싶은 위치(queries)를 입력받아 최종 Intensity 값을 예측
    def decode(self, x, queries):
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        return self.to_outputs(latents)

    # encode, learn, decode를 순서대로 호출하며, 메모리 효율성을 위해 디코딩 시 블록 처리를 수행
    def forward(self, pc, queries, block_size=100000):
        bottleneck = self.encode(pc)
        x = self.learn(bottleneck["x"])

        # 학습 시 query는 전체 grid의 subset. Inference 시에는 전체 grid
        if queries.shape[1] > block_size:
            N = block_size
            os = []
            for block_idx in range(math.ceil(queries.shape[1] / N)):
                o = self.decode(x, queries[:, block_idx * N : (block_idx + 1) * N, :]).squeeze(-1)
                os.append(o)
            o = torch.cat(os, dim=1)
        else:
            o = self.decode(x, queries).squeeze(-1)

        return {"o": o, **bottleneck}


def create_autoencoder(depth=24, dim=512, M=512, N=2048, query_type="point", bottleneck=None, bottleneck_args={}):
    model = VecSetAutoEncoder(
        depth=depth,
        dim=dim,
        output_dim=1,
        num_inputs=N,
        num_latents=M,
        query_type=query_type,
        bottleneck=bottleneck,
        bottleneck_args=bottleneck_args,
    )
    return model


def learnable_vec1024x16_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24,
        dim=1024,
        M=1024,
        N=pc_size,
        query_type="learnable",
        bottleneck=NormalizedBottleneck,
        bottleneck_args={"dim": 1024, "latent_dim": 16},
    )


def learnable_vec1024x32_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24,
        dim=1024,
        M=1024,
        N=pc_size,
        query_type="learnable",
        bottleneck=NormalizedBottleneck,
        bottleneck_args={"dim": 1024, "latent_dim": 32},
    )


def point_vec1024x16_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24,
        dim=1024,
        M=1024,
        N=pc_size,
        query_type="point",
        bottleneck=NormalizedBottleneck,
        bottleneck_args={"dim": 1024, "latent_dim": 16},
    )


def point_vec1024x32_dim1024_depth24_nb(pc_size=8192):
    return create_autoencoder(
        depth=24,
        dim=1024,
        M=1024,
        N=pc_size,
        query_type="point",
        bottleneck=NormalizedBottleneck,
        bottleneck_args={"dim": 1024, "latent_dim": 32},
    )


def learnable_vec1024_dim1024_depth24(pc_size=8192):
    return create_autoencoder(
        depth=24,
        dim=1024,
        M=1024,
        N=pc_size,
        query_type="learnable",
        bottleneck=Bottleneck,
        bottleneck_args={},
    )


def point_vec1024_dim1024_depth24(pc_size=8192):
    return create_autoencoder(
        depth=24,
        dim=1024,
        M=1024,
        N=pc_size,
        query_type="point",
        bottleneck=Bottleneck,
        bottleneck_args={},
    )
