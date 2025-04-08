from abc import ABC, abstractmethod

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr
        
    def update(self, grads):
        num_layers = sum(1 for key in self.parameters.keys() if key.startswith('W'))
    
        for l in range(1, num_layers + 1):
            self.parameters[f'W{l}'] -= self.lr * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.lr * grads[f'db{l}']
            
            if f'dgamma{l}' in grads:
                self.parameters[f'gamma{l}'] -= self.lr * grads[f'dgamma{l}']
                self.parameters[f'beta{l}'] -= self.lr * grads[f'dbeta{l}']

class Scheduler(ABC):
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step(self):
        pass

class StepLR(Scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size  
        self.gamma = gamma          

    def step(self) -> None:
        self.step_count += 1
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma

class MultiStepLR(Scheduler):
    def __init__(self, optimizer, milestones=[200, 500, 900], gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = sorted(milestones)  
        self.gamma = gamma
        self.current_idx = 0 

    def step(self) -> None:
        self.step_count += 1
        if (self.current_idx < len(self.milestones) and 
            self.step_count >= self.milestones[self.current_idx]):
            self.optimizer.lr *= self.gamma
            self.current_idx += 1

class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma=0.95) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.lr *= self.gamma