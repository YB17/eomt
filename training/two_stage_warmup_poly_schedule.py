# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from torch.optim.lr_scheduler import LRScheduler

'''
双阶段 Warmup：先让 head（新加层） warmup，再打开 backbone 的训练，并给 backbone 设定自己的 warmup 时长。
层级保护：backbone 在 head 达到一定稳定后才开始微调。
统一多项式衰减：两个阶段结束后，head/backbone 都进入自己的多项式衰减期，最后 lr → 0。
这样既加快了 head 的收敛，也保留了对预训练 backbone 的稳定保护，提升整体训练效率与最终性能。
'''
class TwoStageWarmupPolySchedule(LRScheduler):
    def __init__(
        self,
        optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        poly_power: float,
        last_epoch=-1,
    ):
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.poly_power = poly_power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        non_vit_warmup, vit_warmup = self.warmup_steps
        for i, base_lr in enumerate(self.base_lrs):
            if i >= self.num_backbone_params:
                if non_vit_warmup > 0 and step < non_vit_warmup:
                    lr = base_lr * (step / non_vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            else:
                if step < non_vit_warmup:
                    lr = 0
                elif step < non_vit_warmup + vit_warmup:
                    lr = base_lr * ((step - non_vit_warmup) / vit_warmup)
                else:
                    adjusted = max(0, step - non_vit_warmup - vit_warmup)
                    max_steps = max(1, self.total_steps - non_vit_warmup - vit_warmup)
                    lr = base_lr * (1 - (adjusted / max_steps)) ** self.poly_power
            lrs.append(lr)
        return lrs
