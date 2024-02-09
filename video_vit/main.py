import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from typing import List, Optional, Callable, Tuple
from beartype import beartype

from einops import pack, unpack, repeat, rearrange
from einops.layers.torch import Rearrange, Reduce


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


# sinusoidal positions


def posemb_sincos_1d(
    seq, dim, temperature=10000, device=None, dtype=torch.float32
):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor(
                (x.shape[0], 1, 1, 1), device=device
            ).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.0,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
        window_size=7,
        num_mem_kv=4,
    ):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.mem_kv = nn.Parameter(
            torch.randn(2, self.heads, num_mem_kv, dim_head)
        )

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1), nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding(
            (2 * window_size - 1) ** 2, self.heads
        )

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        rel_pos += window_size - 1
        rel_pos_indices = (
            rel_pos * torch.tensor([2 * window_size - 1, 1])
        ).sum(dim=-1)

        self.register_buffer(
            "rel_pos_indices", rel_pos_indices, persistent=False
        )

    def forward(self, x):
        (
            batch,
            height,
            width,
            window_height,
            window_width,
            _,
            device,
            h,
        ) = (
            *x.shape,
            x.device,
            self.heads,
        )

        x = self.norm(x)

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h),
            (q, k, v),
        )

        # scale

        q = q * self.scale

        # null / memory / register kv

        mk, mv = map(
            lambda t: repeat(t, "h n d -> b h n d", b=q.shape[0]),
            self.mem_kv,
        )
        num_mem = mk.shape[-2]

        k = torch.cat((mk, k), dim=-2)
        v = torch.cat((mv, v), dim=-2)

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)

        bias = F.pad(bias, (0, 0, num_mem, 0), value=0.0)

        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(
            out,
            "b h (w1 w2) d -> b w1 w2 (h d)",
            w1=window_height,
            w2=window_width,
        )

        # combine heads out

        out = self.to_out(out)
        return rearrange(
            out, "(b x y) ... -> b x y ...", x=height, y=width
        )


class MaxViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3,
    ):
        super().__init__()
        assert isinstance(depth, tuple), (
            "depth needs to be tuple if integers indicating number of"
            " transformer blocks at that stage"
        )

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(
                channels, dim_conv_stem, 3, stride=2, padding=1
            ),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1),
        )

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2**i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        cond_hidden_dims = []

        for ind, (
            (layer_dim_in, layer_dim),
            layer_depth,
        ) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=is_first,
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate,
                    ),
                    Rearrange(
                        "b d (x w1) (y w2) -> b x y w1 w2 d",
                        w1=w,
                        w2=w,
                    ),  # block-like attention
                    Residual(
                        Attention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(
                        FeedForward(dim=layer_dim, dropout=dropout)
                    ),
                    Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
                    Rearrange(
                        "b d (w1 x) (w2 y) -> b x y w1 w2 d",
                        w1=w,
                        w2=w,
                    ),  # grid-like attention
                    Residual(
                        Attention(
                            dim=layer_dim,
                            dim_head=dim_head,
                            dropout=dropout,
                            window_size=w,
                        )
                    ),
                    Residual(
                        FeedForward(dim=layer_dim, dropout=dropout)
                    ),
                    Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
                )

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce("b d h w -> b d", "mean"),
            LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob=0.0,
        return_embeddings=False,
    ):
        x = self.conv_stem(x)

        cond_fns = iter(default(cond_fns, []))

        for stage in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)


class VideoViT(nn.Module):
    """
    VideoViT module for video classification using Vision Transformer (ViT).

    Args:
        num_classes (int): Number of output classes.
        dim (int): Dimension of the token embeddings.
        depth (int): Number of transformer layers.
        dim_head (int, optional): Dimension of the token head embeddings. Defaults to 32.
        dim_conv_stem (int, optional): Dimension of the convolutional stem. Defaults to None.
        window_size (int, optional): Size of the window for local attention. Defaults to 7.
        mbconv_expansion_rate (int, optional): Expansion rate for the mobile inverted bottleneck convolution. Defaults to 4.
        mbconv_shrinkage_rate (float, optional): Shrinkage rate for the mobile inverted bottleneck convolution. Defaults to 0.25.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        channels (int, optional): Number of input channels. Defaults to 3.
    """

    def __init__(
        self,
        num_classes,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=7,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels=3,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.dim_conv_stem = dim_conv_stem
        self.window_size = window_size
        self.mbconv_expansion_rate = mbconv_expansion_rate
        self.mbconv_shrinkage_rate = mbconv_shrinkage_rate
        self.dropout = dropout
        self.channels = channels

        self.vit = MaxViT(
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            dim_conv_stem=dim_conv_stem,
            window_size=window_size,
            mbconv_expansion_rate=mbconv_expansion_rate,
            mbconv_shrinkage_rate=mbconv_shrinkage_rate,
            dropout=dropout,
            channels=channels,
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Forward pass of the VideoViT module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, frames, height, width).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: Output tensor of shape (batch_size, tokens, dim).
        """
        video = rearrange(x, "b c f h w -> b f c h w")
        images, packed_shape = pack_one(video, "* c h w")
        tokens = self.vit(
            images, return_embeddings=True, *args, **kwargs
        )
        print(f"tokens shape: {tokens.shape}")

        tokens = unpack_one(tokens, packed_shape, "* c h w")
        return tokens
