import torch
from torch import nn
from math import log2, sqrt
import torch
from torch import nn, einsum
import torch.nn.functional as F
from dalle_pytorch import distributed_utils
from einops import rearrange
from vector_quantize_pytorch import VectorQuantize


def exists(val):
    return val is not None


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def default(val, d):
    return val if exists(val) else d


class ResBlock(nn.Module):
    def __init__(self, chan, use_SiLU=True):
        super().__init__()
        activation = nn.SiLU if use_SiLU else nn.ReLU
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            activation(),
            nn.Conv2d(chan, chan, 3, padding=1),
            activation(),
            nn.Conv2d(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class VQVAE(nn.Module):
    def __init__(
        self,
        image_size=256,
        num_tokens=512,
        codebook_dim=512,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        channels=3,
        smooth_l1_loss=False,
        alpha=10.0,
        normalization=((*((0.5,) * 3), 0), (*((0.5,) * 3), 1)),
        use_SiLU=True,
        commitment_weight=1.0,
        vq_decay=0.8,
    ):
        super().__init__()
        assert log2(image_size).is_integer(), "image size must be a power of 2"
        assert num_layers >= 1, "number of layers must be greater than or equal to 1"
        has_resblocks = num_resnet_blocks > 0
        activation = nn.SiLU if use_SiLU else nn.ReLU
        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.alpha = alpha

        # self.codebook = nn.Embedding(num_tokens, codebook_dim)
        self.vq = VectorQuantize(
            dim=codebook_dim,
            codebook_size=num_tokens,
            decay=vq_decay,
            commitment_weight=commitment_weight,
            use_cosine_sim=True,
            learnable_codebook=True,
            ema_update=False,
            accept_image_fmap=True,
        )

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(
            lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans)
        )

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), activation()
                )
            )
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1),
                    activation(),
                )
            )

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1], use_SiLU))
            enc_layers.append(ResBlock(enc_chans[-1], use_SiLU))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        # enc_layers.append(nn.Conv2d(enc_chans[-1], codebook_dim, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

        """
            NOTE: Heatmap is already normalized. By removing self.normalization, the code 
            will return the image itself.
        """
        try:
            self.normalization = tuple(map(lambda t: t[:channels], normalization))
        except:
            self.normalization = None

        self._register_external_parameters()

    def _register_external_parameters(self):
        """Register external parameters for DeepSpeed partitioning."""
        if not distributed_utils.is_distributed or not distributed_utils.using_backend(
            distributed_utils.DeepSpeedBackend
        ):
            return

    def norm(self, images):
        if self.normalization is None:
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, "c -> () c () ()"), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        return logits

    def decode(self, img_seq):
        embeds = self.vq.get_output_from_indices(img_seq)
        images = self.decoder(embeds)
        return images

    def forward(
        self,
        img,
        return_loss=False,
        return_recons=False,
        return_logits=False,
    ):
        assert (
            img.shape[-1] == self.image_size and img.shape[-2] == self.image_size
        ), f"input must have the correct image size {self.image_size}"

        img = self.norm(img)

        encoded_features = self.encoder(img)

        embeds, logits, commit_loss = self.vq(encoded_features)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        out = self.decoder(embeds)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        loss = recon_loss + (commit_loss * self.alpha)

        if not return_recons:
            return loss

        return loss, out.clamp(0, 1), logits
