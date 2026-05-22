import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Usage example:


inp_channels=3,
out_channels=3,
dim=48,
num_blocks=[4, 6, 6, 8],
num_refinement_blocks=4,
heads=[1, 2, 4, 8],
ffn_expansion_factor=2.66,
bias=False,
LayerNorm_type='BiasFree',
                                             

self.latent = nn.Sequential(*[
            CondFormerBlock(dim=int(dim * 2 ** 3), z_dim=dim, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])


"""


##########################################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


##########################################################################
## Layer Norm

def to_3d(x):
    # x: (B, C, H, W) -> (B, H*W, C)
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b, h * w, c)

def to_4d(x, h, w):
    # x: (B, H*W, C) -> (B, C, H, W)
    b, hw, c = x.shape
    assert hw == h * w, "Mismatch between provided h*w and input shape"
    return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)

class LFM_layer(nn.Module): # Linearly Fusion Module
    def __init__(self, dim, z_dim):
        super(LFM_layer, self).__init__()
        self.modulation = nn.Sequential(
            nn.Conv2d(dim + z_dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        )
        self.z_dim = z_dim

    def forward(self, x, z):
        '''
        :param x: feature map: B * Cx * H * W
        :param z: prior representation: B * Cz
        '''
        b, c, h, w = x.shape
        z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, self.z_dim, h, w)
        y = torch.cat((x, z), dim=1)

        return self.modulation(y)


class CondAttention(nn.Module):
    def __init__(self, dim, z_dim, num_heads, bias):
        super(CondAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = LFM_layer(dim, z_dim)
        self.k = LFM_layer(dim, z_dim)
        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, z):

        q, k, v = self.q(x, z[:, 0:1]), self.k(x, z[:, 1:2]), self.v(x)

        b, c, h, w = v.shape

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################


class CondFormerBlock(nn.Module):
    def __init__(self, dim, z_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(CondFormerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = CondAttention(dim, z_dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, inp):
        x = inp['x']
        x = x + self.attn(self.norm1(x), inp['z'])
        inp['x'] = x + self.ffn(self.norm2(x))
        return inp





