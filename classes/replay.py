import random
import tensorflow as tf

class Replay:
    
    def __init__(self, cap, batch_size):
        self.memory = []
        self.cap = cap
        self.batch_size = batch_size

    def push(self, exp):
        size = len(self.memory)
        # Remove oldest memory first
        if size == self.cap:
            self.memory.pop(random.randint(0, size-1))
        self.memory.append(exp)
    
    def fetch(self):
        size = len(self.memory)
        # Select batch
        if size < self.batch_size:
            batch = random.sample(self.memory, size)
        else:
            batch = random.sample(self.memory, self.batch_size)
            
        return zip(*batch)