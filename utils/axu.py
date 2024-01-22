import numpy as np
import cv2
import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        args.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        args.gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        args.dist_url = "tcp://%s:%s" % (
            os.environ["MASTER_ADDR"],
            os.environ["MASTER_PORT"],
        )
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["RANK"] = str(args.rank)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}, gpu {}".format(
            args.rank, args.dist_url, args.gpu
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def convert_to_rgb_3d(array, offset=10):
    """
    Convert a 3D numpy array (T, H, W) to a 3D RGB representation (T, 3, H, W).

    Parameters:
    array (numpy.ndarray): Input array of shape (N, T, H, W).
    colormap (str): The name of the colormap to use.

    Returns:
    numpy.ndarray: Output array of shape (T, 3, H, N * W).
    """
    N, T, H, W = array.shape
    output = np.zeros((T, 3, H, N * W), dtype=np.uint8)

    # Apply colormap to each time step
    for n in range(N):
        for t in range(T):
            # Normalize the heatmap for display
            normalized_heatmap = cv2.normalize(
                array[n][t], None, 0, 255, cv2.NORM_MINMAX
            )

            colored_heatmap = cv2.applyColorMap(
                normalized_heatmap.astype("uint8"), cv2.COLORMAP_JET
            )

            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            colored_heatmap = colored_heatmap.transpose((2, 0, 1))

            output[t, 0:, 0:, n * W : (n + 1) * W] = colored_heatmap

    return output


if __name__ == "__main__":
    # Example usage
    N, T, H, W = 4, 60, 256, 256  # dimensions
    example_array = np.zeros((N, T, H, W))
    for n in range(1, N):
        example_array[n] = np.ones((T, H, W)) * (n / N)  # Gaussian distribution

    rgb_array = convert_to_rgb_3d(example_array).transpose((0, 3, 2, 1))

    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    _, new_W, new_H, _ = rgb_array.shape

    out = cv2.VideoWriter(
        "test_rgb.mp4", cv2.VideoWriter_fourcc(*"DIVX"), 30, (new_W, new_H)
    )

    for i in range(T):
        # Write to video
        out.write(rgb_array[i])

    out.release()
