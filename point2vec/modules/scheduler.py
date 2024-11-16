
import math

def build_scheduler(scheduler_type, **kwargs):
    if scheduler_type == 'linear':
        return LinearScheduler(**kwargs)
    elif scheduler_type == 'cosine':
        return CosineScheduler(**kwargs)
    elif scheduler_type == 'constant':
        return ConstantScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class ValueScheduler:
    def __init__(self, initial_value):
        self.initial_value = initial_value
        self.value = initial_value
        self.step_count = 0

    def step(self):
        """Update the value. This method should be overridden by subclasses."""
        raise NotImplementedError

    def state_dict(self):
        return {
            'value': self.value,
            'step_count': self.step_count
        }

    def load_state_dict(self, state_dict):
        self.value = state_dict['value']
        self.step_count = state_dict['step_count']

class LinearScheduler(ValueScheduler):
    def __init__(self, start_value, end_value, total_steps):
        super().__init__(start_value)
        self.end_value = end_value
        self.total_steps = total_steps
        self.delta = (end_value - start_value) / total_steps

    def step(self):
        if self.step_count < self.total_steps:
            self.value += self.delta
        else:
            self.value = self.end_value
        self.step_count += 1


class CosineScheduler(ValueScheduler):
    def __init__(self, start_value, end_value, total_steps):
        super().__init__(start_value)
        self.end_value = end_value
        self.total_steps = total_steps

    def step(self):
        if self.step_count <= self.total_steps:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * self.step_count / self.total_steps))
            self.value = self.end_value + (self.initial_value - self.end_value) * cosine_decay
        else:
            self.value = self.end_value
        self.step_count += 1

class ConstantScheduler(ValueScheduler):
    def __init__(self, constant_value):
        super().__init__(constant_value)

    def step(self):
        # Value remains constant
        self.step_count += 1
