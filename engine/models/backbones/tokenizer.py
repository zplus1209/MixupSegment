import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, C]
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.drop_path1(x) + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path2(x) + h
        return x


# ---------------------------
# Conv Tokenizer
# ---------------------------

class ConvTokenizer(nn.Module):
    """
    Biến ảnh -> lưới token bằng Conv stride=patch_size.
    Dùng positional embedding 2D học được (sin-cos có thể thay).
    """
    def __init__(self, in_ch=3, embed_dim=384, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        # stem -> giữ nhiều texture/biên dạng nội soi
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        # tokenizer conv: stride = patch_size/4 (do stem đã giảm 4x)
        self.tok = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=max(1, patch_size // 4), padding=1, bias=False)

        self.pos_embed = None  # sẽ khởi tạo lazy theo kích thước đầu vào

    def _build_pos_embed(self, H, W, C, device):
        # H,W sau tokenizer
        pe = torch.zeros(1, H * W, C, device=device)
        nn.init.trunc_normal_(pe, std=0.02)
        return nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        # x: [B,3,H,W]
        x = self.stem(x)     # [B,C,H/4,W/4]
        x = self.tok(x)      # [B,C,H',W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C], N = H'*W'

        if self.pos_embed is None or self.pos_embed.shape[1] != (H * W) or self.pos_embed.shape[2] != C:
            self.pos_embed = self._build_pos_embed(H, W, C, x.device)

        x = x + self.pos_embed
        return x  # [B, N, C]


# ---------------------------
# Backbone 1: ConvTokenizerTiny
# ---------------------------

class ConvTokenizerBackbone(nn.Module):
    """
    ConvTokenizer + vài block Transformer.
    Có .classifier Linear để tương thích get_backbone(castrate=True).
    """
    def __init__(self,
                 num_classes: int = 1000,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 patch_size: int = 8,
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 global_pool: str = "mean"):
        super().__init__()
        self.tokenizer = ConvTokenizer(in_ch=3, embed_dim=embed_dim, patch_size=patch_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                             attn_dropout=attn_dropout, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.global_pool = global_pool
        self.classifier = nn.Linear(embed_dim, num_classes)

    @property
    def out_dim(self):
        # tiện nếu bạn muốn đọc trực tiếp
        return self.classifier.in_features

    def forward_features(self, x):
        x = self.tokenizer(x)      # [B, N, C]
        x = self.blocks(x)         # [B, N, C]
        x = self.norm(x)
        if self.global_pool == "mean":
            x = x.mean(dim=1)      # [B, C]
        elif self.global_pool == "cls":
            # lấy token đầu như cls (không thêm cls token ở đây, nên dùng mean là hợp lý hơn)
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        logits = self.classifier(feat)
        return logits


# ---------------------------
# Backbone 2: ConvTokenizerSmall (mạnh hơn)
# ---------------------------

class ConvTokenizerBackboneSmall(ConvTokenizerBackbone):
    def __init__(self, num_classes=1000):
        super().__init__(
            num_classes=num_classes,
            embed_dim=512,     # lớn hơn
            depth=8,           # nhiều block hơn
            num_heads=8,       # nhiều heads hơn
            mlp_ratio=4.0,
            patch_size=8,
            attn_dropout=0.0,
            dropout=0.0,
            global_pool="mean",
        )


# ---------------------------
# Khối MBConv rất gọn (hybrid CNN)
# ---------------------------

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, stride=1):
        super().__init__()
        mid = in_ch * expansion
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.block = nn.Sequential(
            # pw-expand
            nn.Conv2d(in_ch, mid, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            # dw
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            # pw-proj
            nn.Conv2d(mid, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        h = x
        x = self.block(x)
        if self.use_res:
            x = x + h
        return x


# ---------------------------
# Backbone 3: HybridConvViT (CNN stem + Tokenizer + Transformer)
# ---------------------------

class HybridConvViT(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 stem_channels=(32, 64, 128),
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 patch_stride: int = 2,   # stride để tạo token sau stem
                 global_pool: str = "mean"):
        super().__init__()
        # CNN stem (nhẹ, giữ texture nội soi)
        chs = [3] + list(stem_channels)
        stages = []
        for i in range(len(stem_channels)):
            stride = 2 if i == 0 else 2  # giảm kích thước 2x mỗi stage
            stages += [
                MBConv(chs[i], chs[i+1], expansion=4, stride=stride),
                MBConv(chs[i+1], chs[i+1], expansion=4, stride=1),
            ]
        self.stem = nn.Sequential(*stages)  # ra feature map C=stem_channels[-1]

        # map -> embed_dim rồi tokenizer bằng conv stride
        self.to_embed = nn.Conv2d(stem_channels[-1], embed_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.token_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_stride, padding=1, bias=False)

        self.pos_embed = None

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.global_pool = global_pool
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _build_pos(self, H, W, C, device):
        pe = torch.zeros(1, H * W, C, device=device)
        nn.init.trunc_normal_(pe, std=0.02)
        return nn.Parameter(pe, requires_grad=True)

    @property
    def out_dim(self):
        return self.classifier.in_features

    def forward_features(self, x):
        x = self.stem(x)                # [B, C, H', W']
        x = self.to_embed(x)            # [B, E, H', W']
        x = self.token_conv(x)          # [B, E, Ht, Wt]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]

        if self.pos_embed is None or self.pos_embed.shape[1] != (H * W) or self.pos_embed.shape[2] != C:
            self.pos_embed = self._build_pos(H, W, C, x.device)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        if self.global_pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x.mean(dim=1)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        return self.classifier(feat)


# ---------------------------
# Factory functions (để eval("name") hoạt động)
# ---------------------------

def conv_tokenizer_tiny(num_classes: int = 1000):
    return ConvTokenizerBackbone(num_classes=num_classes,
                                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0,
                                 patch_size=8, attn_dropout=0.0, dropout=0.0, global_pool="mean")

def conv_tokenizer_small(num_classes: int = 1000):
    return ConvTokenizerBackboneSmall(num_classes=num_classes)

def hybrid_convvit_tiny(num_classes: int = 1000):
    return HybridConvViT(num_classes=num_classes,
                         stem_channels=(32, 64, 128),  # có thể tăng 32,64,160 cho bản mạnh hơn
                         embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0,
                         patch_stride=2, global_pool="mean")
