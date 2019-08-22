import numpy as np
from processing import get_preprocessed_state, get_postprocessed_state

class Replay(object):

    def __init__(self, cfg, action_size):
        self.cfg = cfg
        # Numpy arrays 5X+ more memory efficient than python lists
        self.obs = np.empty((cfg['cap'], 64, 64, 3), dtype=np.uint8)
        self.actions = np.empty((cfg['cap'], action_size), dtype=np.float32)
        self.rewards = np.empty((cfg['cap'], ), dtype=np.float32) 
        self.nonterminals = np.empty((cfg['cap'], 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def append(self, obs, action, reward, done):
        # Remove oldest memory first
        if self.idx == self.cfg['cap']:
            self.idx = 0
        # Decentre and discretise visual observations (to save memory)
        self.obs[self.idx] = get_postprocessed_state(obs, self.cfg['bit_depth'])
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.nonterminals[self.idx] = not done
        self.idx += 1
        if self.idx == self.cfg['cap']:
            self.full = True
    
    # Returns an index for a valid single sequence chunk uniformly sampled from the memory
    def sample_idxs(self):
        while True:
            # Select start index of chunk
            if self.full:
                idx = np.random.randint(0, self.cfg['cap'])
            else:
                idx = np.random.randint(0, self.idx - self.cfg['chunk_size'])
            # Create list of indexes
            idxs = np.arange(idx, idx + self.cfg['chunk_size']) % self.cfg['cap']
            # Make sure data does not cross the memory index
            if not self.idx in idxs[1:]:
                break

        return idxs
    
    def sample(self):
        # Shape = (batch_size, chunk_size)
        idxs = np.asarray([self.sample_idxs() for _ in range(self.cfg['batch_size'])])
        # Shape = (batch_size * chunk_size)
        vec_idxs = idxs.flatten('F')
        obs = self.obs[vec_idxs].astype(np.float32)
        obs = get_preprocessed_state(obs, self.cfg['bit_depth'])
        
        obs = obs.reshape((self.cfg['chunk_size'], self.cfg['batch_size'], *obs.shape[1:]))
        actions = self.actions[vec_idxs].reshape(self.cfg['chunk_size'], self.cfg['batch_size'], -1)
        rewards = self.rewards[vec_idxs].reshape(self.cfg['chunk_size'], self.cfg['batch_size'])
        nonterminals = self.nonterminals[vec_idxs].reshape(self.cfg['chunk_size'], self.cfg['batch_size'], 1)

        return obs, actions, rewards, nonterminals
