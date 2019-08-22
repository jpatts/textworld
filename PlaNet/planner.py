import tensorflow as tf


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner():

  def __init__(self, cfg, action_size, transition_model, reward_model):
    super().__init__()
    self.cfg = cfg
    self.transition_model = transition_model
    self.reward_model = reward_model
    self.action_size = action_size

  def forward(self, belief, state):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief = tf.expand_dims(belief, axis=1)
    belief = tf.tile(belief, [B, self.cfg['candidates'], H])
    belief = tf.reshape(belief, [-1, H])
    state = tf.expand_dims(state, axis=1)
    state = tf.tile(state, [B, self.cfg['candidates'], Z])
    state = tf.reshape(belief, [-1, Z])
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    action_mean = tf.zeros([self.cfg['planning_horizon'], B, 1, self.action_size])
    action_std_dev = tf.ones([self.cfg['planning_horizon'], B, 1, self.action_size])
    # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
    for _ in range(self.cfg['optimisation_iters']):
        # Sample actions (time x (batch x candidates) x actions)
        actions = action_mean + action_std_dev * tf.random_normal([self.cfg['planning_horizon'], B, self.cfg['candidates'], self.action_size])
        actions = tf.reshape(actions, [self.cfg['planning_horizon'], B * self.cfg['candidates'], self.action_size])
        # Sample next states
        beliefs, states, _, _ = self.transition_model(state, actions, belief)
        # Calculate expected returns (technically sum of rewards over planning horizon)
        beliefs = tf.reshape(beliefs, [-1, H])
        states = tf.reshape(states, [-1, Z])
        rewards = self.reward_model(beliefs, states)
        rewards = tf.reshape(rewards, [self.cfg['planning_horizon'], -1])
        returns = tf.reduce_sum(rewards, axis=0)
        # Re-fit belief to the K best action sequences
        returns = tf.reshape(returns, [B, self.cfg['candidates']])
        _, topk = tf.math.top_k(returns, self.cfg['top_candidates'], sorted=False)
        # Fix indices for unrolled actions
        topk += self.cfg['candidates'] * tf.expand_dims(tf.range(0, B, dtype=tf.int64), axis=1)
        best_actions = tf.reshape(actions[:, tf.reshape(topk, [-1])], [self.cfg['planning_horizon'], B, self.cfg['top_candidates'], self.action_size])
        # Update belief with new means and standard deviations
        action_mean = tf.math.reduce_mean(best_actions, axis=2, keepdims=True)
        action_std_dev = tf.math.reduce_std(best_actions, axis=2, keepdims=True)

    # Return first action mean µ_t
    return tf.squeeze(action_mean[0], axis=1)
