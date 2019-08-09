import spacy, re, glob, gym, textworld.gym
import numpy as np
import tensorflow as tf
from textworld import EnvInfos

class TextWorld(object):

    def __init__(self, cfg):
        super(TextWorld, self).__init__()

        self.cfg = cfg

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

        # Get list of games
        self.games = glob.glob(cfg['data_dir'] + '*.ulx')

        # Start session
        requested_infos = EnvInfos(extras=['walkthrough'])
        requested_infos.entities = True
        requested_infos.admissible_commands = True
        env_id = textworld.gym.register_games(self.games, requested_infos, max_episode_steps=cfg['max_steps'])
        env_id = textworld.gym.make_batch(env_id, batch_size=cfg['num_agents'], parallel=False)
        self.env = gym.make(env_id)

    def reset(self):
        return self.env.reset()
    
    def step(self, commands):
        return self.env.step(commands)
    
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
        
        return np.array(target), tf.one_hot(np.array(target), depth=self.cfg['vocab_size']).numpy(), np.array(target_words)
    
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
    
    # start_entity_end_pad_format, ex: [start, oven, end, pad, pad, pad, start, red, apple, end, pad, pad, start, ....]
    def get_preprocessed_state(self, entities, batch_size, num_entities, max_entity_len):
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
    
    def vec_to_words(self, predictions):
        commands = []
        for batch in predictions:
            final = ''
            for id_ in batch:
                if id_ != self.pad and id_ != self.start and id_ != self.end:
                    final += self.vocab[id_] + ' '
            commands.append(final.strip())
        
        return commands
    
    def get_preprocessed_state2(self, entities, num_entities):
        batched_curr_entities = []
        for _, batch in enumerate(entities):
            curr_entities = [self.start]
            # Create (num_entities, max_entity_len) size list
            for _, id_ in batch.items():
                # If single word
                if type(id_) is int:
                    curr_entities.append(id_)
                # If multiple words
                else:
                    for i in id_:
                        curr_entities.append(i)

            curr_entities.append(self.end)
            # Pad to max number of entities
            for p in range(num_entities - len(curr_entities)):
                curr_entities.append(self.pad)
            # Add to batch
            batched_curr_entities.append(curr_entities)

        return np.array(batched_curr_entities)
