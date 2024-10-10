import torch
from easydict import EasyDict as edict
import yaml

from dalle_pytorch import DiscreteVAE
from dalle_pytorch import Discrete3DVAE
from pose_quant.vqgan import VQGAN


def get_dVAE(config, using_deepspeed=False):
    if config.DATASET.Heatmap_Generator.joint_reduction:
        vae = DiscreteVAE(**config.VAE_Params)
    else:
        vae = Discrete3DVAE(**config.VAE_Params)

    if config.Pretrained_Models.discrete_vae_weight_path != "":
        state_dict = torch.load(
            config.Pretrained_Models.discrete_vae_weight_path, map_location="cpu"
        )["weights"]
        vae.load_state_dict(state_dict)

    return vae


def get_VQ_GAN(config, using_deepspeed=False):
    config_vq_gan = None
    with open(config.VQ_Params.vq_gan_congif) as f:
        config_vq_gan = edict(yaml.load(f, Loader=yaml.FullLoader))

    vq_gan = VQGAN(**config_vq_gan.architecture.vqgan)
    if config_vq_gan.resume:
        vq_gan.load_checkpoint(config_vq_gan.resume)

    return vq_gan
