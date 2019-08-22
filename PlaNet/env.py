import gym, cv2
import numpy as np
from processing import get_preprocessed_state


class GymEnv(object):

    def __init__(self, cfg):
        super(GymEnv, self).__init__()
        self.cfg = cfg
        # Create environment
        self.env = gym.make(cfg['env'])
        # Set random seed
        self.env.seed(cfg['seed'])

        self.action_size = self.env.action_space.shape[0]
    
    def get_obs(self):
        frames = self.env.render(mode='rgb_array')
        images = cv2.resize(frames, (64, 64), interpolation=cv2.INTER_LINEAR)
        images = np.expand_dims(images, axis=0)
        return get_preprocessed_state(images, self.cfg['bit_depth'])

    def reset(self):
        # Reset internal timer
        self.t = 0
        _ = self.env.reset()
        # Get obs
        return self.get_obs()
    
    def step(self):
        total_reward = 0
        step = 0
        action = self.env.action_space.sample()
        # Perform action N times repeatedly
        for k in range(self.cfg['skiprate']):
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            self.t += 1
            if done or self.t == self.cfg['max_steps']:
                break

        return self.get_obs(), action, total_reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
