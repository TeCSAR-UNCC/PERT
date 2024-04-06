import numpy as np
import pickle
import json
import random
import math

from torch.utils.data import Dataset
from ucla_meta import data_dict_val, data_dict_train
from utils.heatmap_related import GeneratePoseTarget

# Heavily inspired by: https://raw.githubusercontent.com/ZhouYuxuanYX/Hyperformer/main/feeders/feeder_ucla.py


class UCAL(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        nw_ucla_root,
        cfg,
        repeat=1,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=-1,
        normalization=False,
        debug=False,
        use_mmap=True,
    ):

        self.is_training = False if "val" in label_path else True

        if "val" in label_path:
            self.train_val = "val"
            self.data_dict = data_dict_val
        else:
            self.train_val = "train"
            self.data_dict = data_dict_train
        self.nw_ucla_root = nw_ucla_root  # "data/NW-UCLA/all_sqe/"
        self.time_steps = 64
        self.bone = [
            (1, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (5, 3),
            (6, 5),
            (7, 6),
            (8, 7),
            (9, 3),
            (10, 9),
            (11, 10),
            (12, 11),
            (13, 1),
            (14, 13),
            (15, 14),
            (16, 15),
            (17, 1),
            (18, 17),
            (19, 18),
            (20, 19),
        ]
        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info["label"]) - 1)

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        self.heatmap_generator = GeneratePoseTarget(
            **cfg.DATASET.Heatmap_Generator,
            skeletons=self.bone,
            left_kp=LEFT_LIMB,
            left_limb=LEFT_LIMB,
            right_kp=RIGHT_LIMB,
            right_limb=RIGHT_LIMB,
            is_training=self.is_training,
        )

        self.load_data()
        # if normalization:
        #    self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        self.data = []
        for data in self.data_dict:
            file_name = data["file_name"]
            with open(self.nw_ucla_root + file_name + ".json", "r") as f:
                json_file = json.load(f)
            skeletons = json_file["skeletons"]
            value = np.array(skeletons)
            self.data.append(value)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = (
            data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        )
        self.std_map = (
            data.transpose((0, 2, 4, 1, 3))
            .reshape((N * T * M, C * V))
            .std(axis=0)
            .reshape((C, 1, V, 1))
        )

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray(
            [
                [1, 0, 0],
                [0, math.cos(agx), math.sin(agx)],
                [0, -math.sin(agx), math.cos(agx)],
            ]
        )
        Ry = np.asarray(
            [
                [math.cos(agy), 0, -math.sin(agy)],
                [0, 1, 0],
                [math.sin(agy), 0, math.cos(agy)],
            ]
        )
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.train_val == "train":

            data = np.zeros((self.time_steps, 20, 3))

            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length)) * 100, self.time_steps)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]

        else:

            data = np.zeros((self.time_steps, 20, 3))

            # value = scalerValue[:, :, :]
            length = value.shape[0]

            idx = np.linspace(0, length - 1, self.time_steps).astype(np.int32)
            data[:, :, :] = value[idx, :, :]  # T,V,C

        if "bone" in self.data_path:
            data_bone = np.zeros_like(data)
            for bone_idx in range(20):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = (
                    data[:, self.bone[bone_idx][0] - 1, :]
                    - data[:, self.bone[bone_idx][1] - 1, :]
                )
            data_bone[:, 2, :] = data[:, 2, :]
            data = data_bone

        ## for joint modality
        ## separate trajectory from relative coordinate to each frame's spine center
        # else:
        # # there's a freedom to choose the direction of local coordinate axes!
        #    trajectory = data[:, 2]
        # let spine of each frame be the joint coordinate center
        #    data = data - data[:, 2:3]
        #
        # ## works well with bone, but has negative effect with joint and distance gate
        #    data[:, 2] = trajectory

        if "motion" in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        # data = np.transpose(data, (2, 0, 1))
        # C, T, V = data.shape
        # data = np.reshape(data, (C, T, V, 1))

        return data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":
    ds = UCAL(
        label_path="train",
        data_path="joint",
        nw_ucla_root="/mnt/DATASETS_RAID_28TB/UCAL/all_sqe/",
    )

    hands = [7, 6, 5, 4, 2, 8, 9, 10, 11]
    legs = [15, 14, 13, 12, 0, 16, 17, 18, 19]
    trunk = [0, 1, 2, 3]

    sample = random.randint(0, len(ds))
    data = ds[sample][0]
    frame = data.shape[1]

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

    for i in range(0, frame):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        # ax = Axes3D(fig, azim=-10, elev=5)

        ax.scatter(data[i, :, 0], data[i, :, 2], data[i, :, 1], c="red", s=40.0)
        ax.plot(
            data[i, hands, 0], data[i, hands, 2], data[i, hands, 1], c="green", lw=2.0
        )
        ax.plot(data[i, legs, 0], data[i, legs, 2], data[i, legs, 1], c="green", lw=2.0)
        ax.plot(
            data[i, trunk, 0], data[i, trunk, 2], data[i, trunk, 1], c="green", lw=2.0
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ax.set_xlim(-0.2, 0.4)
        # ax.set_ylim(2.0, 3.2)
        # ax.set_zlim(-1.0, 1.0)
        ax.view_init(azim=45, elev=30)
        plt.savefig("./test_fig/idx_{}_frame{}.png".format(sample, i))

    print()
