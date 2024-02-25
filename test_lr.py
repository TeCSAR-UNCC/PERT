import torch
from torch.optim import Adam
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from torchvision.models.mobilenetv2 import MobileNetV2
import matplotlib.pyplot as plt

model = MobileNetV2()

opt = Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.05,
)

dl_size = 1000

epoch = 40

step_size_up = int(5 * dl_size)

sched = WarmupCosineLR(
    optimizer=opt,
    total_num_steps=epoch * dl_size,
    warmup_min_ratio=0,
    warmup_num_steps=step_size_up,
    cos_min_ratio=1e-8,
)

lrs = []
for _ in range(epoch):
    for _ in range(dl_size):
        sched.step()
        lr = sched.get_last_lr()
        lrs.append(lr)

plt.plot(lrs)
plt.savefig("test_lr.png")
