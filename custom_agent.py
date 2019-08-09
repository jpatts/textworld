import random, yaml
import numpy as np
import tensorflow as tf
from tqdm import trange
from classes.textworld_api import TextWorld
from classes.models import Seq2seq
from classes.replay import Replay
from classes.writer import Writer


tf.compat.v1.enable_eager_execution()
# Load cfg
with open('cfg.yaml') as reader:
    cfg = yaml.safe_load(reader)

class CustomAgent:

    def __init__(self):
        self.textworld = TextWorld(cfg)
        self.replay = Replay(cfg['cap'], cfg['batch_size'])
        self.writer = Writer(cfg['extension'], cfg['save_dir'], cfg['log_dir'], cfg['log_freq'])
        self.model = Seq2seq(cfg['vocab_size'], cfg['embedding_size'], cfg['units'], cfg['dropout'])
        self.optim = tf.keras.optimizers.Adam(cfg['learning_rate'])
    
    #@tf.function()
    def update(self):
        # Problem: Experiences with different max #entities are being sampled, creating unusable matrix
        entities, teachers, scores = self.replay.fetch()

        # Max length of each entity (+2 is for start and end tokens)
        cmd_len = max([len(batch) for batch in teachers])

        # Pad teachers
        for batch in teachers:
            for _ in range(cmd_len - len(batch)):
                batch.append(self.textworld.pad)

        x_in = self.textworld.get_preprocessed_state2(entities)
        teachers = np.array(teachers)
        one_hot_targets = tf.one_hot(teachers, depth=cfg['vocab_size'])

        # Compute gradients
        with tf.GradientTape() as tape:
            logits, predictions = self.model(x_in, cmd_len, teachers, cfg['batch_size'])
            #print(self.textworld.vec_to_words(predictions))
            #print()

            # Compare target and predicted
            loss = tf.compat.v1.losses.softmax_cross_entropy(one_hot_targets, logits, reduction='none')
            #loss = tf.losses.huber_loss(one_hot_targets, logits)
            # Apply mask to remove pad gradients
            mask = tf.cast(tf.math.logical_not(tf.math.equal(teachers, 0)), dtype=tf.dtypes.float32)
            loss = loss * mask
            loss = tf.reduce_mean(loss + scores) 

        # Log
        self.writer.log(self.optim, tape, loss)
        self.writer.global_step.assign_add(1)

        # Calculate and apply gradients
        grads = tape.gradient(loss, self.model.weights)
        grads_and_vars = zip(grads, self.model.weights)
        self.optim.apply_gradients(grads_and_vars)

    #@tf.function
    def train(self):
        for epoch in trange(1, cfg['epochs']):
            # Init scraped entities list
            entities = []
            for b in range(cfg['num_agents']):
                entities.append({})
            
            # Get initial obs, info from game
            obs, infos = self.textworld.reset()

            # Get valid entities that exist in game
            true_entities = infos['entities']
            # Get solutions to all current batches 
            walkthroughs = infos['extra.walkthrough']

            # Max length of each entity (+2 is for start and end tokens)
            cmd_len = 2 + max([ max([len(cmd.split()) for cmd in batch]) for batch in walkthroughs])

            target, one_hot_target, target_words = self.textworld.get_targets(walkthroughs, cmd_len)
            # Get starting index for each batch of target
            curr_targets = [[x for x in range(cfg['num_agents'])], [0 for x in range(cfg['num_agents'])]]
            # Init loop vars
            dones = [False] * cfg['num_agents']
            scores = [0] * cfg['num_agents']
            memories = []
            while not all(dones):
                # Update global entity list
                entities = self.textworld.update_entities(entities, true_entities, obs)
                # Get input data
                x_in = self.textworld.get_preprocessed_state2(entities)
                teachers = target[tuple(curr_targets)]
                # Run through model
                logits, predictions = self.model(x_in, cmd_len, teachers, cfg['num_agents'])
                # Ignore padded chars
                mask1 = tf.cast(tf.math.logical_not(tf.math.equal(one_hot_target[tuple(curr_targets)], -float("Inf"))), dtype=tf.dtypes.float32)
                logits = logits * mask1
                mask2 = tf.cast(tf.math.logical_not(tf.math.equal(target[tuple(curr_targets)], 0)), dtype=tf.dtypes.int64)
                predictions = predictions * mask2
                
                # Convert from ids to words
                commands = self.textworld.vec_to_words(predictions)

                print(commands, target_words[tuple(curr_targets)])
                
                # Perform commands
                obs, scores, dones, infos = self.textworld.step(commands)

                # Add to replay buffer
                for b in range(cfg['num_agents']):
                    teacher = [item for item in teachers[b].tolist() if item != 0]
                    #id_ = hash(" ".join(str(x) for x in teacher))
                    # Save entities, solution without <PAD> tokens, target for logits
                    memories.append([entities[b], teacher])

                # Increment target indexes for successful batch commands
                correct = tf.cast(tf.math.equal(target[tuple(curr_targets)], predictions), dtype=tf.dtypes.float32)
                #print()
                for b, batch in enumerate(correct):
                    # if 1 for each column
                    if int(tf.reduce_sum(input_tensor=tf.abs(batch))) == cmd_len:
                        #print("CORRECT!")
                        curr_targets[1][b] += 1

            # Add end-game score to memories, add memories to experience replay
            for i in range(len(memories)):
                memories[i].append(scores)
                self.replay.push(memories[i])
            if len(self.replay.memory) >= cfg['batch_size']:
                self.update()

def main():
    model = CustomAgent()
    model.train()

if __name__ == '__main__':
    main()
