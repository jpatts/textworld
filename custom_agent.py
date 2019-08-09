import random, yaml, re, spacy, gym
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

        # Init models
        self.textworld = TextWorld(cfg['data_dir'], cfg['max_steps'], cfg['num_agents'])
        self.replay = Replay(cfg['cap'], cfg['batch_size'])
        self.writer = Writer(cfg['extension'], cfg['save_dir'], cfg['log_dir'], cfg['log_freq'])
        self.model = Seq2seq(cfg['vocab_size'], cfg['embedding_size'], cfg['units'])
        self.optim = tf.keras.optimizers.Adam(cfg['learning_rate'])

        # Load vocab
        self.nlp = spacy.load('en')
        with open('./vocab.txt') as f:
            self.vocab = f.read().split('\n')

        # Create tokenized dict
        self.word2id = {}
        for i, w in enumerate(self.vocab):
            self.word2id[w] = i

        # Set important tokens
        self.start = self.word2id['<S>']
        self.pad = self.word2id['<PAD>']
        self.end = self.word2id['</S>']
        self.unk = self.word2id['<UNK>']

        self.preposition_map = {'chop':'with', 'cook':'with', 'dice':'with', 'insert':'into', 'lock':'with',
                                'put':'on', 'slice':'with', 'take':'from', 'unlock':'with'}
        self.preposition_map_encoded = {self.word2id[key]: self.word2id[val] for (key, val) in self.preposition_map.items()}
        
        self.possible_cmds = ['chop', 'close', 'cook', 'dice', 'drink', 'drop', 'eat', 'examine', 'go', 'insert',
                              'inventory', 'lock', 'look', 'open', 'prepare', 'put', 'slice', 'take', 'unlock']
        self.possible_cmds_encoded = [self.word2id[cmd] for cmd in self.possible_cmds]
    
    def get_targets(self, walkthrough, max_cmd_len):
        target = []
        target_words = []
        # For each batch
        for batch in walkthrough:
            # Add initial commands that we want agent to perform?
            #examine_cookbook = [self.start, self.word2id['examine'], self.word2id['cookbook'], self.end]
            target_batch = []
            target_words_batch = []
            # For each command
            for cmd in batch:
                # Add start token
                target_cmd = [self.start]
                target_words_cmd = ['<S>']
                # Add correct words
                for word in cmd.split():
                    target_cmd.append(self.word2id[word.lower()])
                    target_words_cmd.append(word.lower())
                # Add end token
                target_cmd.append(self.end)
                target_words_cmd.append('</S>')
                # Pad if necessary
                for _ in range(max_cmd_len - len(target_cmd)):
                    target_cmd.append(self.pad)
                    target_words_cmd.append('<PAD>')
                # Append to structured list
                target_batch.append(target_cmd)
                target_words_batch.append(target_words_cmd)
            target.append(target_batch)
            target_words.append(target_words_batch)
        
        return np.array(target), tf.one_hot(np.array(target), depth=cfg['vocab_size']).numpy(), np.array(target_words)
    
    def update_entities(self, entities, true_entities, obs):
        # Loop through batches
        for b, batch in enumerate(obs):
            # Remove unwanted chars
            to_remove = ['\\n', '\\', '|', '$', '/', '>', '_']
            for c in to_remove:
                batch = batch.replace(c, ' ')

            # Remove multiple whitespaces
            batch = re.sub(' +', ' ', batch)
            # Remove everything between '-=' and '=-'
            batch = re.sub('-=.+?=-', '', batch)

            doc = self.nlp(batch)
            # Get all noun phrases
            for np in doc.noun_chunks:
                # Get nouns and adjective-noun pairs; reason for both is that 'closed stove' is not equal to 'stove' in textworld
                noun = ''
                adj_noun = ''
                for token in np:
                    # Thing is a pronoun; I don't know why spacy considers it a noun, so it must be removed
                    if token.pos_ is 'NOUN' and 'thing' not in token.text:
                        noun += token.text.lower() + ' '
                        adj_noun += token.text.lower() + ' '
                    elif token.pos_ is 'ADJ':
                        adj_noun += token.text.lower() + ' '
                
                # Remove trailing whitespace
                noun = noun.strip()
                adj_noun = adj_noun.strip()
                # If valid, unique and not empty, convert to id and add to entity dict
                if noun != '' and noun not in entities[b] and noun in true_entities[b]:
                    n_id = [self.word2id[n] for n in noun.split()]
                    entities[b][noun] = n_id
                if adj_noun != '' and adj_noun not in entities[b] and adj_noun in true_entities:
                    adj_id = [self.word2id[adj] for adj in adj_noun.split()]
                    entities[b][adj_noun] = adj_id
        return entities
            
    # max_len = 4, ex: [start, oven, end, pad, pad, pad, start, red, apple, end, pad, pad, start, ....]
    def get_start_entity_end_pad_format(self, entities, batch_size, num_entities, max_entity_len):
        batched_curr_entities = []
        for _, batch in enumerate(entities):
            curr_entities = []
            # Create (num_entities, max_entity_len) size list
            for _, id_ in batch.items():
                entity = [self.start]
                # If single word
                if type(id_) is int:
                    entity.append(id_)
                # If multiple words
                else:
                    for i in id_:
                        entity.append(i)

                entity.append(self.end)
                # Pad entity to max length
                for p in range(max_entity_len - len(entity)):
                    entity.append(self.pad)
                # Add to valid entities
                curr_entities.append(entity)
        
            # Pad to max number of entities
            for p in range(num_entities - len(curr_entities)):
                padding = [self.pad] * max_entity_len
                curr_entities.append(padding)
            # Collapse dimensions for training
            curr_entities = np.reshape(np.array(curr_entities), (num_entities * max_entity_len))
            # Add to batch
            batched_curr_entities.append(curr_entities)

        batched_curr_entities = np.reshape(np.array(batched_curr_entities), [batch_size, num_entities * max_entity_len])
        return batched_curr_entities
    
    #@tf.function()
    def update(self):
        # Compute gradients
        with tf.GradientTape() as tape:
            # Problem: Experiences with different max #entities are being sampled, creating unusable matrix
            entities, max_entity_len, num_entities, max_cmd_len, target, teacher, scores = self.replay.fetch()
            self.model.encoder.reset_hidden(cfg['batch_size'])
        
            x_in = self.get_start_entity_end_pad_format(entities, cfg['batch_size'], max(num_entities), max(max_entity_len))
            target = np.array(target)
            teacher = np.array(teacher)
            scores = np.array(scores)

            logits, predictions = self.model(x_in, max(max_cmd_len), teacher)

            # Compare target and predicted
            loss = tf.compat.v1.losses.softmax_cross_entropy(target, logits, reduction='none')
            #loss = tf.losses.mean_squared_error(scores[0] + 0.99 * target[0], logits)
            # Apply mask to remove pad gradients
            mask = tf.cast(tf.math.logical_not(tf.math.equal(teacher, 0)), dtype=tf.dtypes.float32)
            loss = loss * mask
            loss = 1/2 * tf.reduce_mean(input_tensor=loss)

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
            
            target, one_hot_target, target_words = self.get_targets(walkthrough, max_cmd_len)
            # Get starting index for each batch of target
            curr_targets = [[x for x in range(cfg['num_agents'])], [0 for x in range(cfg['num_agents'])]]
            # Init loop vars
            dones = [False] * cfg['num_agents']
            scores = [0] * cfg['num_agents']
            memories = []
            while not all(dones):
                # Update global entity list
                entities = self.update_entities(entities, true_entities, obs)
                # Get input data
                x_in = self.get_start_entity_end_pad_format(entities, cfg['num_agents'], num_entities, max_entity_len)
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
                commands = []
                for batch in predictions:
                    final = ''
                    for id_ in batch:
                        if id_ != self.pad and id_ != self.start and id_ != self.end:
                            final += self.vocab[id_] + ' '
                    commands.append(final.strip())
                
                # Perform commands
                obs, scores, dones, infos = self.textworld.step(commands)
                print("commands")
                print(commands)
                print()
                #print("targets")
                #print(target_words[tuple(curr_targets)])
                #print()
                #score = np.reshape(score, (cfg['num_agents'], max_cmd_len, cfg['vocab_size']))

                # Add to replay buffer
                for b in range(cfg['num_agents']):
                    memories.append([entities[b], max_entity_len, num_entities, max_cmd_len, one_hot_target[tuple(curr_targets)][b], teacher[b]])

                # Increment target indexes for successful batch commands
                correct = tf.cast(tf.math.equal(target[tuple(curr_targets)], predictions), dtype=tf.dtypes.float32)
                print("correct")
                print(correct)
                print()
                #print()
                for b, batch in enumerate(correct):
                    # if 1 for each column
                    if int(tf.reduce_sum(input_tensor=tf.abs(batch))) == max_cmd_len:
                        curr_targets[1][b] += 1

            # Add end-game score to memories, add memories to experience replay
            for i in range(len(memories)):
                memories[i].append(scores)
                self.replay.push(memories[i])
            self.update()

def main():
    model = CustomAgent()
    model.train()

if __name__ == '__main__':
    main()
