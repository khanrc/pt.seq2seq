# Below code is modified from https://github.com/ildoonet/pytorch-gradual-warmup-lr
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer, which has base_lr.
        init_scale: init_lr = base_lr * init_scale
        T_max: warmup steps
        after: after T_max steps, use this scheduler
    """

    def __init__(self, optimizer, init_scale, T_max, after=None):
        self.scale = init_scale
        self.T_max = T_max
        self.after = after
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.T_max:
            if self.after:
                if not self.finished:
                    self.finished = True
                return self.after.get_lr()
            else:
                return self.base_lrs

        progress = self.last_epoch / self.T_max
        return [(base_lr - base_lr*self.scale) * progress + base_lr*self.scale
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after:
            return self.after.step(epoch)
        else:
            return super().step(epoch)


if __name__ == "__main__":
    # Test run
    import torch.nn as nn
    import torch.optim as optim

    optimizer = optim.Adam([nn.Parameter()], lr=3e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=3e-6)
    scheduler = WarmupLR(optimizer, init_scale=0.01, T_max=10, after=scheduler)

    for epoch in range(40):
        lr = scheduler.get_lr()
        print(epoch, lr)
        scheduler.step()
