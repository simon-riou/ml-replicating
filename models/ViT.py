import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()

        self.linear_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        self.flatten = nn.Flatten(2, 3)

    def forward(self, x):
        x = self.flatten(self.linear_proj(x))
        return torch.permute(x, (0, 2, 1))
  
class ClassificationHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(ClassificationHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=in_features
            ),
            nn.GELU(),
            nn.Linear(
                in_features=in_features,
                out_features=out_features
            ),
        )

    def forward(self, x):
        return self.mlp(x)
    
class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadSelfAttentionBlock, self).__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_output, _  = self.mha(query=x_norm, key=x_norm, value=x_norm)
        return attn_output + x
    
class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio):
        super(MLPBlock, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=mlp_ratio*embed_dim),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(in_features=mlp_ratio*embed_dim, out_features=embed_dim),
            nn.Dropout(0.1),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.mlp(self.norm(x)) + x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoderBlock, self).__init__()

        self.multihead_self_attention_block = MultiheadSelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)

        self.mlp_block = MLPBlock(embed_dim=embed_dim, mlp_ratio=4)

    def forward(self, x):
        return self.mlp_block(self.multihead_self_attention_block(x))
    
class ViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, nb_blocks, embed_dim, num_heads, out_classes):
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim)

        num_patches = (img_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList()
        for _ in range(nb_blocks):
            self.blocks.append(TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads))

        self.norm = nn.LayerNorm(embed_dim)

        self.classification_head = ClassificationHead(in_features=embed_dim, out_features=out_classes)



    def forward(self, x):
        x = self.patch_embedding(x)
        class_embedding = self.class_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((class_embedding, x), dim=1)
        x = x + self.position_embedding

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return self.classification_head(x[:,0])