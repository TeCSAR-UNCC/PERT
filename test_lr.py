import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torchvision.models.mobilenetv2 import MobileNetV2
import matplotlib.pyplot as plt

model = MobileNetV2()

opt = Adam(
    model.parameters(),
    lr=1e-6,
    weight_decay=0.05,
)

dl_size = 1000

step_size_up = int(2 * dl_size)
step_size_down = int(1 * dl_size)

sched = CyclicLR(
    optimizer=opt,
    base_lr=1e-6,
    max_lr=1e-4,
    mode="exp_range",
    step_size_up=step_size_up,
    step_size_down=step_size_down,
    cycle_momentum=False,
    gamma=0.99991,
)

lrs = []
for _ in range(30):
    for _ in range(dl_size):
        sched.step()
        lrs.append(sched.get_last_lr())

plt.plot(lrs)
plt.savefig("test_lr.png")
