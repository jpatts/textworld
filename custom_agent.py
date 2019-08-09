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
        self.model = Seq2seq(cfg['vocab_size'], cfg['embedding_size'], cfg['units'])
        self.optim = tf.keras.optimizers.Adam(cfg['learning_rate'])
    
    #@tf.function()
    def update(self):
        # Compute gradients
        with tf.GradientTape() as tape:
            self.model.encoder.reset_hidden(cfg['batch_size'])
            # Problem: Experiences with different max #entities are being sampled, creating unusable matrix
            entities, max_entity_len, num_entities, max_cmd_len, target, teacher, scores = self.replay.fetch()
    
            target = np.array(target)
            teacher = np.array(teacher)
            scores = np.array(scores)
        
            x_in = self.textworld.get_preprocessed_state(entities, cfg['batch_size'], max(num_entities), max(max_entity_len))
            logits, predictions = self.model(x_in, max(max_cmd_len), teacher)

            # Compare target and predicted
            loss = tf.compat.v1.losses.softmax_cross_entropy(target, logits, reduction='none')
            #loss = tf.losses.mean_squared_error(scores[0] + 0.99 * target[0], logits)
            # Apply mask to remove pad gradients
            mask = tf.cast(tf.math.logical_not(tf.math.equal(teacher, 0)), dtype=tf.dtypes.float32)
            loss = loss * mask
            loss = 1/2 * tf.reduce_mean(loss + scores) 

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
            self.model.encoder.reset_hidden(cfg['num_agents'])
            # Init scraped entities list
            entities = []
            for b in range(cfg['num_agents']):
                entities.append({})
            
            # Get initial obs, info from game
            obs, infos = self.textworld.reset()

            # Get valid entities that exist in game
            true_entities = infos['entities']
            # Get solutions to all current batches 
            walkthrough = infos['extra.walkthrough']

            # Max length of each entity (+2 is for start and end tokens)
            max_entity_len = 2 + max([ max([len(entity.split()) for entity in batch]) for batch in true_entities])
            # Max length of each command
            max_cmd_len = 2 + 2*(max_entity_len)
            # Total number of entities in game
            num_entities = max([len(batch) for batch in true_entities])
            
            target, one_hot_target, target_words = self.textworld.get_targets(walkthrough, max_cmd_len)
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
                x_in = self.textworld.get_preprocessed_state(entities, cfg['num_agents'], num_entities, max_entity_len)
                #print(x_in.shape)
                teacher = target[tuple(curr_targets)]
                # Run through model
                logits, predictions = self.model(x_in, max_cmd_len, teacher)

                # Ignore padded chars
                mask1 = tf.cast(tf.math.logical_not(tf.math.equal(one_hot_target[tuple(curr_targets)], -float("Inf"))), dtype=tf.dtypes.float32)
                logits = logits * mask1
                mask2 = tf.cast(tf.math.logical_not(tf.math.equal(target[tuple(curr_targets)], 0)), dtype=tf.dtypes.int64)
                predictions = predictions * mask2
                
                # Convert from ids to words
                commands = self.textworld.vec_to_words(predictions)
                
                # Perform commands
                obs, scores, dones, infos = self.textworld.step(commands)
                print("commands")
                print(commands)
                print()
                print("targets")
                print(target_words[tuple(curr_targets)][0])
                print()
                #score = np.reshape(score, (cfg['num_agents'], max_cmd_len, cfg['vocab_size']))

                # Add to replay buffer
                for b in range(cfg['num_agents']):
                    memories.append([entities[b], max_entity_len, num_entities, max_cmd_len, one_hot_target[tuple(curr_targets)][b], teacher[b]])

                # Increment target indexes for successful batch commands
                correct = tf.cast(tf.math.equal(target[tuple(curr_targets)], predictions), dtype=tf.dtypes.float32)
                print(correct)
                #print()
                for b, batch in enumerate(correct):
                    # if 1 for each column
                    if int(tf.reduce_sum(input_tensor=tf.abs(batch))) == max_cmd_len:
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
