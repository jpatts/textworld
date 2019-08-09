import glob, gym, textworld.gym
from textworld import EnvInfos

class TextWorld(object):

    def __init__(self, data_dir, max_steps, num_agents):
        super(TextWorld, self).__init__()

        # Get list of games
        self.games = glob.glob(data_dir + '*.ulx')

        # Start session
        requested_infos = EnvInfos(extras=['walkthrough'])
        requested_infos.entities = True
        requested_infos.admissible_commands = True
        env_id = textworld.gym.register_games(self.games, requested_infos, max_episode_steps=max_steps)
        env_id = textworld.gym.make_batch(env_id, batch_size=num_agents, parallel=False)
        self.env = gym.make(env_id)

    def reset(self):
        return self.env.reset()
    
    def step(self, commands):
        return self.env.step(commands)