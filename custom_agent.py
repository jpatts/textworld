import random, glob, yaml, re, spacy, gym, textworld.gym
import numpy as np
import tensorflow as tf
from tqdm import trange
from textworld import EnvInfos
from models import Seq2seq
from utils.writer import Writer

tf.enable_eager_execution()
# Load cfg
with open('cfg.yaml') as reader:
    cfg = yaml.safe_load(reader)

class Replay:
    
    def __init__(self):
        self.memory = []
        self.cap = cfg["replay"]["cap"]
        self.batch_size = cfg["train"]["batch_size"]

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


class CustomAgent:

    def __init__(self):

        self.nlp = spacy.load('en')
        # Load vocab
        with open('./vocab.txt') as f:
            self.vocab = f.read().split('\n')
            #self.vocab = ' '.join(self.vocab)
            #self.vocab = get_adj_nouns(self.nlp, [], self.vocab)

        # Create tokenized dict
        self.word2id = {}
        for i, w in enumerate(self.vocab):
            self.word2id[w] = i

        # Important tokens
        self.start = self.word2id['<S>']
        self.pad = self.word2id['<PAD>']
        self.end = self.word2id['</S>']
        self.unk = self.word2id['<UNK>']
        # List of games
        self.games = glob.glob(cfg['general']['data_dir'] + '*.ulx')

        self.vocab_size = cfg['vocab']['size']
        self.embedding_size = cfg['model']['embedding_size']
        self.batch_size = cfg['train']['batch_size']
        self.units = cfg['model']['units']
        self.model = Seq2seq(self.vocab_size, self.embedding_size, self.batch_size, self.units)
        self.optim = tf.train.AdamOptimizer(cfg['train']['learning_rate'])
        self.replay = Replay()

        # Start session
        requested_infos = EnvInfos(extras=['walkthrough'])
        requested_infos.entities = True
        requested_infos.admissible_commands = True
        env_id = textworld.gym.register_games(self.games, requested_infos, max_episode_steps=cfg['train']['max_steps'])
        env_id = textworld.gym.make_batch(env_id, batch_size=self.batch_size, parallel=False)
        self.env = gym.make(env_id)
        
        self.preposition_map = {'chop':'with', 'cook':'with', 'dice':'with', 'insert':'into', 'lock':'with', 'put':'on', 'slice':'with', 'take':'from', 'unlock':'with'}
        self.possible_cmds = ['chop', 'close', 'cook', 'dice', 'drink', 'drop', 'eat', 'examine', 'go', 'insert',
                                'inventory', 'lock', 'look', 'open', 'prepare', 'put', 'slice', 'take', 'unlock']
        self.possible_cmds_encoded = [self.word2id[cmd] for cmd in self.possible_cmds]
        self.preposition_map_encoded = {self.word2id[key]: self.word2id[val] for (key, val) in self.preposition_map.items()}
        self.epoch = tf.Variable(0)
        self.writer = Writer(cfg['writing']['extension'], cfg['writing']['save_dir'], cfg['writing']['log_dir'], cfg['writing']['log_freq'])
    
    def update_entities(self, obs):
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
                if noun != '' and noun not in self.entities[b] and noun in self.true_entities[b]:
                    n_id = [self.word2id[n] for n in noun.split()]
                    self.entities[b][noun] = n_id
                if adj_noun != '' and adj_noun not in self.entities[b] and adj_noun in self.true_entities:
                    adj_id = [self.word2id[adj] for adj in adj_noun.split()]
                    self.entities[b][adj_noun] = adj_id
            
    # max_len = 4, ex: [start, oven, end, pad, pad, pad, start, red, apple, end, pad, pad, start, ....]
    def get_start_entity_end_pad_format(self):
        batched_curr_entities = []
        for b, batch in enumerate(self.entities):
            curr_entities = []
            # Create (num_entities, max_entity_len) size list
            for name, id_ in batch.items():
                entity = [self.start]
                # If single word
                if type(id_) is int:
                    entity.append(self.id_)
                # If multiple words
                else:
                    for i in id_:
                        entity.append(i)

                entity.append(self.end)
                # Pad entity to max length
                for p in range(self.max_entity_len - len(entity)):
                    entity.append(self.pad)
                # Add to valid entities
                curr_entities.append(entity)
        
            # Pad to max number of entities
            for p in range(self.num_entities - len(curr_entities)):
                padding = [self.pad] * self.max_entity_len
                curr_entities.append(padding)
            # Collapse dimensions for training
            curr_entities = np.reshape(np.array(curr_entities), (self.num_entities * self.max_entity_len))
            # Add to batch
            batched_curr_entities.append(curr_entities)

        batched_curr_entities = tf.reshape(np.array(batched_curr_entities), [self.batch_size, self.num_entities * self.max_entity_len])
        return batched_curr_entities
            
    # max_len = 4, ex: [start, oven, pad, pad, pad, pad, red, apple, pad, pad, pad, pad, fridge, ...., end, pad, pad, .... pad]
    def get_start_pad_entity_pad_end_format(self):
        # Start, end + (entity + space_pad) * N
        max_len = 2 + (self.max_entity_len + 1) * self.num_entities
        batched_curr_entities = []
        # For each batch
        for b, batch in enumerate(self.entities):
            # Add batch start token
            curr_entities = [self.start]
            # For each entity
            for name, id_ in batch.items():
                count = 0
                # Add single word
                if type(id_) is int:
                    count += 1
                    curr_entities.append(self.id_)
                # Add multiple words
                else:
                    for i in id_:
                        count += 1
                        curr_entities.append(i)
                
                # Pad to max length, add extra pad as space
                for p in range(self.max_entity_len - count):
                    curr_entities.append(self.pad)
            # Add batch end token
            curr_entities.append(self.end)
            # Pad to max number of entities
            for p in range(max_len - len(curr_entities)):
                curr_entities.append(self.pad)
            # Add to batch
            batched_curr_entities.append(curr_entities)
        
        batched_curr_entities = tf.reshape(np.array(batched_curr_entities), [self.batch_size, max_len])
        return batched_curr_entities
    
    def update(self):
        # Compute gradients
        with tf.GradientTape() as tape:
            x_in, target, teacher, scores = self.replay.fetch()
            logits, _ = self.model(x_in[0], self.max_cmd_len, teacher[0])

            # Compare target and predicted
            loss = tf.losses.softmax_cross_entropy(target[0], logits, reduction='none')
            #loss = 1/2 * tf.reduce_mean(tf.losses.huber_loss(score + self.discount * one_hot_target[curr_targets], logits))
            # Apply mask to remove pad gradients
            mask = tf.cast(tf.math.logical_not(tf.math.equal(teacher[0], 0)), dtype=tf.dtypes.float32)
            loss = loss * mask
            loss = tf.reduce_mean(loss)

            # Log
            self.writer.log(self.optim, tape, loss)
            self.writer.global_step.assign_add(1)
            
            # Calculate and apply gradients
            grads = tape.gradient(loss, self.model.weights)
            grads_and_vars = zip(grads, self.model.weights)
            self.optim.apply_gradients(grads_and_vars)

    def train(self):
        for epoch in trange(1, cfg['train']['epochs']):
            for game in range(len(self.games)):
                # Init scraped entities list
                self.entities = []
                for b in range(self.batch_size):
                    self.entities.append({})
                
                # Get initial obs, info from game
                obs, infos = self.env.reset()

                # Get valid entities that exist in game
                self.true_entities = infos['entities']
                # Max length of each entity (+2 is for start and end tokens)
                self.max_entity_len = 2 + max([ max([len(entity.split()) for entity in batch]) for batch in self.true_entities])
                # Max length of each command
                self.max_cmd_len = 2 + 2*(self.max_entity_len)
                # Total number of entities in game
                self.num_entities = max([len(batch) for batch in self.true_entities])

                # Get solutions to all current batches 
                walkthrough = infos['extra.walkthrough']
                target = []
                # For each batch
                for batch in walkthrough:
                    # Add initial commands that we want agent to perform?
                    examine_cookbook = [self.start, self.word2id['examine'], self.word2id['cookbook'], self.end]
                    target_batch = []
                    # For each command
                    for cmd in batch:
                        # Add start token
                        target_cmd = [self.start]
                        # Add correct words
                        for word in cmd.split():
                            target_cmd.append(self.word2id[word.lower()])
                        # Add end token
                        target_cmd.append(self.end)
                        # Pad if necessary
                        for _ in range(self.max_cmd_len - len(target_cmd)):
                            target_cmd.append(self.pad)
                        # Append to structured list
                        target_batch.append(target_cmd)
                    target.append(target_batch)

                target = np.array(target)
                one_hot_target = tf.one_hot(target, depth=self.vocab_size).numpy()
                # Get starting index for each batch of target
                curr_targets = [[x for x in range(self.batch_size)], [0 for x in range(self.batch_size)]]
                # Init loop vars
                dones = [False] * self.batch_size
                scores = [0] * self.batch_size
                memories = []
                while not all(dones):
                    # Update global entity list
                    self.update_entities(obs)
                    # Get input data
                    x_in = self.get_start_entity_end_pad_format()
                    teacher = target[tuple(curr_targets)]
                    logits, predictions = self.model(x_in, self.max_cmd_len, teacher)

                    # Ignore padded chars
                    mask1 = tf.cast(tf.math.logical_not(tf.math.equal(one_hot_target[tuple(curr_targets)], 0)), dtype=tf.dtypes.float32)
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
                    obs, scores, dones, infos = self.env.step(commands)
                    print(commands)
                    #score = np.reshape(score, (self.batch_size, self.max_cmd_len, self.vocab_size))
                    memories.append([x_in, one_hot_target[tuple(curr_targets)], teacher])

                    # Increment target indexes for successful batch commands
                    correct = tf.cast(tf.math.equal(target[tuple(curr_targets)], predictions), dtype=tf.dtypes.float32)
                    for b, batch in enumerate(correct):
                        # if 1 for each column
                        if int(tf.reduce_sum(tf.abs(batch))) == self.max_cmd_len:
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
