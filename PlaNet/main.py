import yaml, os
import numpy as np
import tensorflow as tf
from tqdm import trange
from env import GymEnv
from memory import Replay
from models import Transition, Observation, Reward, Encoder
from planner import MPCPlanner


# Remove logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.enable_eager_execution()
# Load cfg
with open('cfg.yaml') as reader:
    cfg = yaml.safe_load(reader)

# Set seeds
np.random.seed(cfg['seed'])
tf.random.set_random_seed(cfg['seed'])

# Initialise training environment and experience replay memory
env = GymEnv(cfg)
replay = Replay(cfg, env.action_size)
# Initialise dataset replay with S random seed episodes
for s in range(cfg['seed_episodes']):
    observation = env.reset()
    done = False
    while not done:
        next_observation, action, reward, done = env.step()
        replay.append(observation, action, reward, done)
        observation = next_observation

# Init PlaNet
transition_model = Transition(cfg)
observation_model = Observation(cfg)
reward_model = Reward(cfg)
encoder = Encoder(cfg)

optim = tf.train.AdamOptimizer(cfg['learning_rate'], epsilon=cfg['optim_eps'])
planner = MPCPlanner(cfg, env.action_size, transition_model, reward_model)
global_prior = tf.distributions.Normal(tf.zeros([cfg['batch_size'], cfg['state_size']]), tf.ones([cfg['batch_size'], cfg['state_size']]))  # Global prior N(0, I)
free_nats = tf.fill(dims=[1, ], value=cfg['free_nats'])  # Allowed deviation in KL divergence

# Training
for episode in trange(cfg['train']['episodes']):
    # Model fitting
    losses = []
    for _ in trange(cfg['collect_interval']):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)}
        obs, actions, rewards, nonterminals = replay.sample()
        # Create initial belief and state for time t = 0
        init_belief = tf.zeros([cfg['batch_size'], cfg['belief_size']])
        init_state = tf.zeros([cfg['batch_size'], cfg['state_size']])

        with tf.GradientTape() as tape:
            obs_in = tf.reshape(obs, [obs.shape[0] * obs.shape[1], *obs.shape[2:]])
            encoded_obs = encoder(obs_in)
            encoded_obs = tf.reshape(encoded_obs, [obs.shape[0], obs.shape[1], *encoded_obs.shape[1:]])
            # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(actions[:-1], init_state, init_belief, encoded_obs, nonterminals[:-1])

            beliefs_in = tf.reshape(beliefs, [beliefs.shape[0] * beliefs.shape[1], *beliefs.shape[2:]])
            posterior_states_in = tf.reshape(posterior_states, [posterior_states.shape[0] * posterior_states.shape[1], *posterior_states.shape[2:]])
            obs_logits = observation_model(beliefs_in, posterior_states_in)
            print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            obs_logits = tf.reshape(obs_logits, [beliefs.shape[0], beliefs.shape[1], *obs_logits.shape[1:]])
            # Calculate observation likelihood
            obs_loss = tf.keras.losses.MSE(obs_logits, obs[1:], REDUCTION='NONE')
            obs_loss = tf.reduce_sum(obs_loss, axis=[2, 3, 4])
            obs_loss = tf.reduce_mean(obs_loss, axis=[0, 1])
            print('sex')
            exit(1)

            # Calculate reward likelihood
            reward_logits = reward_model(beliefs, posterior_states)
            reward_loss = tf.keras.losses.MSE(reward_logits, rewards[:-1], REDUCTION='NONE')
            reward_loss = tf.reduce_mean(reward_loss, axis=[0, 1])

            # Calculate KL losses (for t = 0 only for latent overshooting); sum over final dims
            kl_loss = tf.reduce_sum([Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)], axis=2)
            kl_loss = tf.reduce_max(kl_divergence(kl_loss, cfg['free_nats']))
            kl_loss = tf.reduce_mean(kl_loss, axis=[0, 1])

            # Average over batch and time
            if cfg['global_kl_beta'] != 0:
                kl_loss += cfg['global_kl_beta'] * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0, 1))
            
            # Calculate latent overshooting objective for t > 0
            if cfg['overshooting_kl_beta'] != 0:
                overshooting_vars = []  # Collect variables for overshooting to process in batch
                for t in range(1, cfg['chunk_size'] - 1):
                    d = min(t + cfg['overshooting_distance'], cfg['chunk_size'] - 1)  # Overshooting distance
                    t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
                    seq_pad = (0, 0, 0, 0, 0, t - d + cfg['overshooting_distance'])  # Calculate sequence padding so overshooting terms can be calculated in one batch
                    # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
                    overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, cfg['batch_size'], cfg['state_size']), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
            overshooting_vars = tuple(zip(*overshooting_vars))
            # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
            beliefs, prior_states, prior_means, prior_std_devs = transition_model(tf.concat(overshooting_vars[4], axis=0), tf.concat(overshooting_vars[0], axis=1), tf.concat(overshooting_vars[3], axis=0), None, tf.concat(overshooting_vars[1], axis=1))
            seq_mask = tf.concat(overshooting_vars[7], axis=1)
            # Calculate overshooting KL loss with sequence mask
            kl_loss += (1 / cfg['overshooting_distance']) * cfg['overshooting_kl_beta'] * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (cfg['chunk_size'] - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 
            # Calculate overshooting reward prediction loss with sequence mask
            if cfg['overshooting_reward_scale'] != 0:
                reward_loss += (1 / cfg['overshooting_distance']) * cfg['overshooting_reward_scale'] * F.mse_loss(bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (cfg['chunk_size'] - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 

            total_loss = obs_loss + reward_loss + kl_loss
        
        # Calculate and apply gradients
        grads = tape.gradient(total_loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optim.apply_gradients(grads_and_vars)

    # Data collection
    observation = env.reset()
    belief = tf.zeros(1, cfg['belief_size'])
    posterior_state = tf.zeros(1, cfg['state_size'])
    action = tf.zeros(1, env.action_size)
    for t in tqdm(range(cfg['max_steps'] // cfg['skiprate'])):
        # Infer belief over current state q(s_t|o≤t,a<t) from the history
        # Action and observation need extra time dimension
        belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, tf.expand_dims(action, axis=0), belief, tf.expand_dims(encoder(observation), axis=0))
        # Remove time dimension from belief/state
        belief = tf.squeeze(belief, axis=0)
        posterior_state = tf.squeeze(posterior_state, axis=0)
        # Get action from planner(q(s_t|o≤t,a<t), p)
        action = planner(belief, posterior_state)
        # Add exploration noise ε ~ p(ε) to the action
        action = action + cfg['action_noise'] * tf.random_normal(action.shape)
        # Perform environment step (action repeats handled internally)
        next_observation, reward, done = env.step(action[0])

        replay.append(observation, action, reward, done)
        observation = next_observation
        if cfg['render']:
            env.render()

# Close training environment
env.close()
