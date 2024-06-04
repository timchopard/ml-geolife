import torchvision.models as tvm
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, out_dim * 5)
        self.fc2 = nn.Linear(out_dim * 5, out_dim)
        self.norm = nn.LayerNorm(out_dim * 5)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.norm(x)
        x = self.fc2(x)
        return x


class Multihead(nn.Module):
    def __init__(self, heads, head_dim, dim):
        super().__init__()
        self.head_dim = head_dim
        self.heads = heads
        self.inner_dim = self.heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3)
        self.to_output = nn.Linear(self.inner_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        Q, K, V = map(
            lambda t: rearrange(t, "b l (h dim) -> b h l dim", dim=self.head_dim), qkv
        )
        K_T = K.transpose(-1, -2)
        att_score = Q @ K_T * self.scale
        att = self.softmax(att_score)
        out = att @ V  # (B,H,L,dim)
        out = rearrange(out, "b h l dim -> b l (h dim)")
        output = self.to_output(out)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, heads, head_dim, out_dim):
        super().__init__()
        self.MHA = Multihead(heads=heads, head_dim=head_dim, dim=dim)
        # self.FeedForward = MLP(dim=dim, out_dim=out_dim)
        self.FeedForward = FeedForward(dim=dim, mlp_dim=out_dim)

    def forward(self, x):
        x = self.MHA(x) + x
        x = self.FeedForward(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, num_class):
        super().__init__()
        self.transformer = Transformer(
            dim=dim, heads=heads, head_dim=head_dim, out_dim=mlp_dim
        )

        self.MLP_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_class))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        CLS_token = x[:, 0, :]
        out = self.MLP_head(CLS_token)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()

        self.resnet18 = tvm.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(
            4, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet18.maxpool = nn.Identity()
        self.ln = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc1(x)
        return x
