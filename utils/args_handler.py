import argparse
from dalle_pytorch import distributed_utils
from configs.config import update_config

# argument parsing
parser = argparse.ArgumentParser()
parser = distributed_utils.wrap_arg_parser(parser)


def get_args():
    parser = argparse.ArgumentParser("PERT pre-training script", add_help=False)
    config_group = parser.add_argument_group("Config")
    config_group.add_argument(
        "--cfg",
        default="configs/PERT/pert.yaml",
        help="experiment configure file name",
        required=False,
        type=str,
    )

    model_group = parser.add_argument_group("Training settings")

    model_group.add_argument(
        "--model",
        default="beit_base_patch16_224_8k_vocab",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    model_group.add_argument("--rel_pos_bias", action="store_true")
    model_group.add_argument(
        "--disable_rel_pos_bias", action="store_false", dest="rel_pos_bias"
    )
    model_group.set_defaults(rel_pos_bias=True)
    model_group.add_argument("--abs_pos_emb", action="store_true")
    model_group.set_defaults(abs_pos_emb=False)
    model_group.add_argument(
        "--layer_scale_init_value",
        default=0.1,
        type=float,
        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale",
    )

    # Note: Added to config file
    """
    model_group.add_argument(
        "--num_mask_patches",
        default=75,
        type=int,
        help="number of the visual tokens/patches need be masked",
    )
    model_group.add_argument("--max_mask_patches_per_block", type=int, default=None)
    model_group.add_argument("--min_mask_patches_per_block", type=int, default=16)
    """

    model_group.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )
    parser.add_argument(
        "--second_interpolation",
        type=str,
        default="lanczos",
        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")',
    )

    """
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    """

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    return parser.parse_args()
