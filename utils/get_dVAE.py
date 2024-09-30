import torch
from dalle_pytorch import DiscreteVAE
from dalle_pytorch import Discrete3DVAE


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
