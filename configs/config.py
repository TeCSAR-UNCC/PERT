# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()


def _update_dict(k, v):
    if k not in config.keys():
        config[k] = edict()

    for vk, vv in v.items():
        config[k][vk] = vv


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if isinstance(v, dict):
                _update_dict(k, v)
                continue
            config[k] = v


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def get_model_name(cfg):
    name = "{model}".format(model=cfg.MODEL)
    full_name = "TEST"

    return name, full_name


if __name__ == "__main__":
    import sys

    update_config(sys.argv[1])

    print(config)
