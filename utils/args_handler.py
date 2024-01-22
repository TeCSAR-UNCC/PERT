import argparse
from dalle_pytorch import distributed_utils

# argument parsing
parser = argparse.ArgumentParser()
parser = distributed_utils.wrap_arg_parser(parser)


def get_args():
    parser = argparse.ArgumentParser("PERT pre-training script", add_help=False)
    config_group = parser.add_argument_group("Config")
    config_group.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    config_group.add_argument(
        "--heatmap_size", type=int, required=False, default=256, help="image size"
    )

    parser = distributed_utils.wrap_arg_parser(parser)

    train_group = parser.add_argument_group("Training settings")

    train_group.add_argument("--epochs", type=int, default=20, help="number of epochs")

    train_group.add_argument("--batch_size", type=int, default=8, help="batch size")

    train_group.add_argument(
        "--base_learning_rate", type=float, default=1e-6, help="learning rate"
    )

    train_group.add_argument(
        "--max_learning_rate", type=float, default=4e-4, help="Minimum learning rate"
    )

    train_group.add_argument(
        "--weight_decay", type=float, default=1e-1, help="learning rate decay"
    )

    # FIXME: Not used any more!
    train_group.add_argument(
        "--lr_decay_rate", type=float, default=0.99, help="learning rate decay"
    )

    train_group.add_argument(
        "--starting_temp", type=float, default=1.0, help="starting temperature"
    )

    train_group.add_argument(
        "--temp_min", type=float, default=0.5, help="minimum temperature to anneal to"
    )

    train_group.add_argument(
        "--anneal_rate", type=float, default=1e-6, help="temperature annealing rate"
    )

    dVAE_model_group = parser.add_argument_group("dVAE Model settings")
    dVAE_model_group.add_argument("--discrete_vae_weight_path", type=str)
    dVAE_model_group.add_argument(
        "--num_tokens", type=int, default=8192, help="number of image tokens"
    )
    dVAE_model_group.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="number of layers (should be 3 or above)",
    )
    dVAE_model_group.add_argument(
        "--num_resnet_blocks", type=int, default=2, help="number of residual net blocks"
    )
    dVAE_model_group.add_argument(
        "--emb_dim", type=int, default=512, help="embedding dimension"
    )
    dVAE_model_group.add_argument(
        "--hidden_dim", type=int, default=256, help="hidden dimension"
    )

    args = parser.parse_args()

    update_config(args.cfg)

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

    model_group.add_argument(
        "--num_mask_patches",
        default=75,
        type=int,
        help="number of the visual tokens/patches need be masked",
    )
    model_group.add_argument("--max_mask_patches_per_block", type=int, default=None)
    model_group.add_argument("--min_mask_patches_per_block", type=int, default=16)

    model_group.add_argument(
        "--input_size", default=256, type=int, help="images input size for backbone"
    )
    model_group.add_argument(
        "--second_input_size",
        default=256,
        type=int,
        help="images input size for discrete vae",
    )

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

    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=10, type=int)
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
