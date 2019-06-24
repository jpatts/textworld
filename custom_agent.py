import glob, yaml, re, spacy, gym, textworld.gym
import numpy as np
import tensorflow as tf
from tqdm import trange
from textworld import EnvInfos
from models import Seq2seq
from utils.utils import get_adj_nouns, get_all_cmds


tf.enable_eager_execution()
# Load cfg
with open('cfg.yaml') as reader:
    cfg = yaml.safe_load(reader)

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
        self.requested_infos = EnvInfos()
        self.requested_infos.entities = True
        self.requested_infos.admissible_commands = True

        env_id = textworld.gym.register_games(self.games, requested_infos, max_episode_steps=cfg['train']['max_steps'])
        env_id = textworld.gym.make_batch(env_id, batch_size=self.batch_size, parallel=False)
        self.env = gym.make(env_id)

        self.vocab_size = cfg['vocab']['size']
        self.embedding_size = cfg['model']['embedding_size']
        self.batch_size = cfg['train']['batch_size']
        self.units = cfg['model']['units']
        self.model = Seq2seq(self.vocab_size, self.embedding_size, self.batch_size, self.units, self.start, self.end)
        self.optim = tf.keras.optimizers.Adam(cfg['train']['learning_rate'])
        
        self.preposition_map = {'chop':'with', 'cook':'with', 'dice':'with', 'insert':'into', 'lock':'with', 'put':'on', 'slice':'with', 'take':'from', 'unlock':'with'}
        self.possible_cmds = ['chop', 'close', 'cook', 'dice', 'drink', 'drop', 'eat', 'examine', 'go', 'insert',
                                'inventory', 'lock', 'look', 'open', 'prepare', 'put', 'slice', 'take', 'unlock']
        self.possible_cmds_encoded = [self.word2id[cmd] for cmd in self.possible_cmds]
        self.preposition_map_encoded = {self.word2id[key]: self.word2id[val] for (key, val) in self.preposition_map.items()}
        self.epoch = tf.Variable(0)

    def get_action(self, obs, true_entities, max_entity_len, num_entities, admissible_commands):

        def update_entities(obs):
            # Loop through batches
            for b, batch in enumerate(obs):
                self.entities.append({})
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
                    if noun != '' and noun not in self.entities[b] and noun in true_entities[b]:
                        n_id = [self.word2id[n] for n in noun.split()]
                        self.entities[b][noun] = n_id
                    if adj_noun != '' and adj_noun not in self.entities[b] and adj_noun in true_entities:
                        adj_id = [self.word2id[adj] for adj in adj_noun.split()]
                        self.entities[b][adj_noun] = adj_id
        
        # max_len = 4, ex: [start, oven, end, pad, pad, pad, start, red, apple, end, pad, pad, start, ....]
        def get_start_entity_end_pad_format():
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

            batched_curr_entities = tf.reshape(np.array(batched_curr_entities), [self.batch_size, num_entities * max_entity_len])
            return batched_curr_entities
        
        # max_len = 4, ex: [start, oven, pad, pad, pad, pad, red, apple, pad, pad, pad, pad, fridge, ...., end, pad, pad, .... pad]
        def get_start_pad_entity_pad_end_format():
            # Start, end + (entity + space_pad) * N
            max_len = 2 + (max_entity_len + 1) * num_entities
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
                    for p in range(max_entity_len - count):
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

        
        # Update global entity list
        update_entities(obs)

        with tf.GradientTape() as tape:
            # Get actions
            action = self.model(get_start_entity_end_pad_format(), max_entity_len)
            # Convert from ids to words
            commands = []
            for batch in action:
                final = ''
                for id_ in batch:
                    if id_ != self.pad and id_ != self.start and id_ != self.end:
                        final += self.vocab[id_] + ' '
                commands.append(final.strip())
            # Perform commands
            obs, score, done, infos = self.env.step(commands)

            for batch in action:
                if batch in admissible_commands:
    
    def train(self):
        for epoch in trange(1, cfg['train']['epochs']):
            for game in range(len(self.games)):
                self.entities = []
                obs, infos = env.reset()

                old_score = np.zeros((self.batch_size))
                done = False
                true_entities = infos['entities']
                # +2 is for start and end tokens
                max_entity_len = max([ max([len(entity.split()) for entity in batch]) for batch in true_entities]) + 2
                num_entities = max([len(batch) for batch in true_entities])
                while not done:
                    admissible_commands = infos['admissible_commands']
                    action = self.perform_action(obs, true_entities, max_entity_len, num_entities, admissible_commands)

def main():
    model = CustomAgent()
    model.train()

if __name__ == '__main__':
    main()
