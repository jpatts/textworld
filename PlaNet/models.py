import tensorflow as tf
import numpy as np


# Takes previous state and possible actions vector, predicts future states
class Transition(tf.keras.Model):

  def __init__(self, cfg):
    super(Transition, self).__init__()
    self.cfg = cfg
    self.embed_state_action = tf.keras.layers.Dense(cfg['belief_size'], activation=cfg['activ'])
    self.rnn = tf.keras.layers.GRUCell(cfg['belief_size'])
    self.embed_belief_prior = tf.keras.layers.Dense(cfg['hidden_size'], activation=cfg['activ'])
    self.state_prior = tf.keras.layers.Dense(2 * cfg['state_size'])
    self.embed_belief_posterior = tf.keras.layers.Dense(cfg['hidden_size'], activation=cfg['activ'])
    self.state_posterior = tf.keras.layers.Dense(2 * cfg['state_size'])

  def call(self, actions, prev_state, prev_belief, observations=None, nonterminals=None):
    T = actions.shape[0] + 1
    # Create tensors for hidden states -> tuple(tf.zeros([T, 1] for i in range(7)))
    beliefs = [np.empty(0)] * T
    prior_states = [np.empty(0)] * T
    prior_means = [np.empty(0)] * T
    prior_std_devs = [np.empty(0)] * T
    posterior_states = [np.empty(0)] * T
    posterior_means = [np.empty(0)] * T
    posterior_std_devs = [np.empty(0)] * T
    # Initialize first indexes
    beliefs[0] = prev_belief
    prior_states[0] = prev_state
    posterior_states[0] = prev_state
    # Loop over time sequence
    for t in range(T - 1):
        # Select appropriate previous state
        state = prior_states[t]
        if observations is not None:
            state = posterior_states[t]

        # Mask if previous transition was terminal
        if nonterminals is not None:
            state = state * nonterminals[t]
        
        # Add action to state
        x_in = tf.concat([state, actions[t]], axis=1)
        # Compute belief (deterministic hidden state)
        belief_hidden = self.embed_state_action(x_in)
        beliefs[t + 1], _ = self.rnn(belief_hidden, [beliefs[t]])
        # Compute state PRIOR by applying transition dynamics
        prior_hidden = self.embed_belief_prior(beliefs[t + 1])
        prior_hidden = self.state_prior(prior_hidden)
        # Split PRIOR into means and standard deviation
        prior_means[t + 1], prior_std_dev = tf.split(prior_hidden, 2, axis=1)
        # Calculate next states standard deviation
        prior_std_devs[t + 1] = tf.keras.activations.softplus(prior_std_dev) + self.cfg['min_std_dev']
        # Infer next state
        prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * tf.random.normal(prior_means[t + 1].shape)     
        if observations is not None:
            # Add belief to observation
            x_in = tf.concat([beliefs[t + 1], observations[t]], axis=1)
            # Compute state POSTERIOR by applying transition dynamics and using current observation
            posterior_hidden = self.embed_belief_posterior(x_in)
            posterior_hidden = self.state_posterior(posterior_hidden)
            # Split POSTERIOR into means and standard deviation
            posterior_means[t + 1], posterior_std_dev = tf.split(posterior_hidden, 2, axis=1)
            # Calculate next states standard deviation
            posterior_std_devs[t + 1] = tf.keras.activations.softplus(posterior_std_dev) + self.cfg['min_std_dev']
            # Infer next state
            posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * tf.random.normal(posterior_means[t + 1].shape)
    # Return new hidden states
    hidden = [tf.stack(beliefs[1:], axis=0), tf.stack(prior_states[1:], axis=0), tf.stack(prior_means[1:], axis=0), tf.stack(prior_std_devs[1:], axis=0)]
    if observations is not None:
      hidden += [tf.stack(posterior_states[1:], axis=0), tf.stack(posterior_means[1:], axis=0), tf.stack(posterior_std_devs[1:], axis=0)]
    return hidden


# Decoder, takes hidden state as input, returns observation (image)
class Observation(tf.keras.Model):
  
  def __init__(self, cfg):
    super(Observation, self).__init__()
    self.cfg = cfg
    self.fc = tf.keras.layers.Dense(cfg['embedding_size'])
    self.convs = tf.keras.Sequential([
        # Filters, Kernel Size, Strides
        tf.keras.layers.Conv2DTranspose(128, 5, 2, activation=cfg['activ']),
        tf.keras.layers.Conv2DTranspose(64, 5, 2, activation=cfg['activ']),
        tf.keras.layers.Conv2DTranspose(32, 6, 2, activation=cfg['activ']),
        tf.keras.layers.Conv2DTranspose(3, 6, 2)
    ])

  def call(self, belief, state):
    x_in = tf.concat([belief, state], axis=1)
    hidden = self.fc(x_in)
    hidden = tf.reshape(hidden, [-1, self.cfg['embedding_size'], 1, 1])
    print(hidden.shape)
    observation = self.convs(hidden)
    return observation


# Returns percieved reward from given state
class Reward(tf.keras.Model):

  def __init__(self, cfg):
    super(Reward, self).__init__()
    self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(cfg['hidden_size'], activation=cfg['activ']),
        tf.keras.layers.Dense(cfg['hidden_size'], activation=cfg['activ']),
        tf.keras.layers.Dense(1),

    ])

  def call(self, belief, state):
    x_in = tf.concat([belief, state], axis=1)
    reward = self.model(x_in)
    return tf.squeeze(reward, axis=1)


# Encodes observation (image) into hidden state
class Encoder(tf.keras.Model):
  
  def __init__(self, cfg):
    super(Encoder, self).__init__()
    self.cfg = cfg
    self.model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 4, 2, activation=cfg['activ']),
        tf.keras.layers.Conv2D(64, 4, 2, activation=cfg['activ']),
        tf.keras.layers.Conv2D(128, 4, 2, activation=cfg['activ']),
        tf.keras.layers.Conv2D(256, 4, 2, activation=cfg['activ']),
    ])
    self.fc = tf.keras.layers.Dense(cfg['embedding_size'])

  def call(self, observation):
    hidden = self.model(observation)
    hidden = tf.reshape(hidden, [-1, 1024])
    # Identity if embedding size is 1024, else linear projection
    if self.cfg['embedding_size'] != 1024:
        return self.fc(hidden)
    else:
        return tf.identity(hidden)
