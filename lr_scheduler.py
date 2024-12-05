from functools import partial
import torch.optim.lr_scheduler as sch

class base():
    def __init__(self, *args):
        self.optimizer = args[0]

    def step(self):
        pass

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
class Warmup(sch._LRScheduler):
    def __init__(self, optimizer, num_annealing_steps, num_total_steps):
        self.num_annealing_steps = num_annealing_steps
        self.num_total_steps = num_total_steps

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_annealing_steps:
            return [base_lr * self._step_count / self.num_annealing_steps for base_lr in self.base_lrs]
        else:
            return self.base_lrs


def get_sch(scheduler, optimizer, **kwargs):
    if scheduler=='None':
        return base(optimizer)
    elif scheduler=='warmup':
        return Warmup(optimizer, num_annealing_steps=kwargs['warmup_epochs'], num_total_steps=kwargs['warmup_epochs'],)
    else:
        NotImplementedError(f'{scheduler} not implemented')