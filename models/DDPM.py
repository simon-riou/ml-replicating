import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm

class TimeEmbedding(nn.Module):
    def __init__(self, dim):

        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):

        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        embeddings = self.mlp(embeddings)

        return embeddings


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_emb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        residual = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual

class SelfAttentionBlock(nn.Module):
    def __init__(self, C):
        super(SelfAttentionBlock, self).__init__()
        
        self.C = C
        self.d = C // 8
        self.scale = self.d ** -0.5
        
        self.q_proj = nn.Conv2d(C, self.d, kernel_size=1)
        self.k_proj = nn.Conv2d(C, self.d, kernel_size=1)
        self.v_proj = nn.Conv2d(C, self.d, kernel_size=1)
        
        self.out_proj = nn.Conv2d(self.d, C, kernel_size=1)
        
        self.norm = nn.GroupNorm(32, C)
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        x = self.norm(x)
        
        # Dim after proj -> [B, d, H, W]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention : [B, d, H*W]
        q = q.view(B, self.d, H * W)
        k = k.view(B, self.d, H * W)
        v = v.view(B, self.d, H * W)
        
        # for matmul -> [B, H*W, d]
        q = q.transpose(1, 2)  
        v = v.transpose(1, 2)
        
        # Compute  attention : [B, H*W, H*W]
        scores = torch.matmul(q, k) * self.scale  # [B, H*W, d] @ [B, d, H*W]
        attention = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)  # [B, H*W, d]
        
        # Reshape back : [B, d, H, W]
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, self.d, H, W)
        
        out = self.out_proj(out)  # [B, C, H, W]
        
        return out + residual

class UNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, out_channels=3,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2, time_emb_dim=128, dropout=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        self.time_embedding = TimeEmbedding(time_emb_dim)
        time_emb_dim_mlp = time_emb_dim * 4

        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        channels = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResNetBlock(ch, out_ch, time_emb_dim_mlp, dropout)
                )
                ch = out_ch
                channels.append(ch)

            if level in [1]:
                self.encoder_blocks.append(
                    SelfAttentionBlock(ch)
                )

            # Downsample
            if level != len(channel_mult) - 1:
                self.downsample_blocks.append(
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
                channels.append(ch)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResNetBlock(ch, ch, time_emb_dim_mlp, dropout),
            ResNetBlock(ch, ch, time_emb_dim_mlp, dropout),
        ])

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                self.decoder_blocks.append(
                    ResNetBlock(ch + skip_ch, out_ch, time_emb_dim_mlp, dropout)
                )
                ch = out_ch

            if level in [3]:
                self.decoder_blocks.append(
                    SelfAttentionBlock(ch)
                )

            # Upsample
            if level != 0:
                self.upsample_blocks.append(
                    nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
                )

        # Final output
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.act_out = nn.ReLU()
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        time_emb = self.time_embedding(t)

        h = self.conv_in(x)

        skip_connections = [h]

        block_idx = 0
        downsample_idx = 0

        for block in self.encoder_blocks:
            if isinstance(block, SelfAttentionBlock):
                h = block(h)
                skip_connections.append(h)
            else:
                h = block(h, time_emb)
                skip_connections.append(h)
                block_idx += 1

                if block_idx % self.num_res_blocks == 0 and downsample_idx < len(self.downsample_blocks):
                    h = self.downsample_blocks[downsample_idx](h)
                    skip_connections.append(h)
                    downsample_idx += 1

        for block in self.bottleneck:
            h = block(h, time_emb)

        block_idx = 0
        upsample_idx = 0

        for block in self.decoder_blocks:
            if isinstance(block, SelfAttentionBlock):
                skip = skip_connections.pop()
                h = h + skip
                h = block(h)
            else:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
                block_idx += 1

                if block_idx % (self.num_res_blocks + 1) == 0 and upsample_idx < len(self.upsample_blocks):
                    h = self.upsample_blocks[upsample_idx](h)
                    upsample_idx += 1

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h


class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_prod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[t]

        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None, None]

        noisy_images = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise

        return noisy_images, noise

    def sample_timesteps(self, batch_size):
        # Uniform disribution
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

    def step(self, model_output, timestep, sample):
        t = timestep

        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

        beta_t = beta_t[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[:, None, None, None]
        sqrt_recip_alpha_t = sqrt_recip_alpha_t[:, None, None, None]

        pred_original_sample = sqrt_recip_alpha_t * (
            sample - beta_t * model_output / sqrt_one_minus_alpha_cumprod_t
        )

        variance = self.posterior_variance[t]
        variance = variance[:, None, None, None]

        noise = torch.randn_like(sample)
        noise[t == 0] = 0

        pred_prev_sample = pred_original_sample + torch.sqrt(variance) * noise

        return pred_prev_sample


class DDPM(nn.Module):
    def __init__(self, in_channels=3, model_channels=128, out_channels=3,
                 channel_mult=(1, 2, 2, 2), num_res_blocks=2, time_emb_dim=256,
                 num_timesteps=1000, beta_start=0.0001, beta_end=0.02, dropout=0.1):
        """
        Args:
            in_channels: Number of input channels
            model_channels: Base channel count for UNet
            out_channels: Number of output channels
            channel_mult: Channel multiplier for each resolution level
            num_res_blocks: Number of residual blocks per level
            time_emb_dim: Time embedding dimension
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta for noise schedule
            beta_end: Ending beta for noise schedule
            dropout: Dropout probability
        """
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            time_emb_dim=time_emb_dim,
            dropout=dropout
        )

        self.num_timesteps = num_timesteps
        self.scheduler = None  # <-- Init in training
        self.beta_start = beta_start
        self.beta_end = beta_end

    def forward(self, x, t):
        return self.unet(x, t)

    def compute_loss(self, x_0, device):
        if self.scheduler is None:
            self.scheduler = DDPMScheduler(
                num_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                device=device
            )

        batch_size = x_0.shape[0]

        t = self.scheduler.sample_timesteps(batch_size)

        noisy_images, noise = self.scheduler.add_noise(x_0, t)

        noise_pred = self.forward(noisy_images, t)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def sample(self, batch_size, image_size, channels=3, device='cuda'):
        if self.scheduler is None:
            self.scheduler = DDPMScheduler(
                num_timesteps=self.num_timesteps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                device=device
            )

        x = torch.randn(batch_size, channels, image_size, image_size, device=device)

        pbar = tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps)
        for t in pbar:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            noise_pred = self.forward(x, t_batch)

            x = self.scheduler.step(noise_pred, t_batch, x)

        return x