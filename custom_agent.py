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

        self.model = Seq2seq(cfg['vocab']['size'], cfg['model']['embedding_size'])
        self.optim = tf.keras.optimizers.Adam(cfg['train']['learning_rate'])
        
        self.preposition_map = {'chop':'with', 'cook':'with', 'dice':'with', 'insert':'into', 'lock':'with', 'put':'on', 'slice':'with', 'take':'from', 'unlock':'with'}
        self.possible_cmds = ['chop', 'close', 'cook', 'dice', 'drink', 'drop', 'eat', 'examine', 'go', 'insert',
                                'inventory', 'lock', 'look', 'open', 'prepare', 'put', 'slice', 'take', 'unlock']
        self.possible_cmds_encoded = [self.word2id[cmd] for cmd in self.possible_cmds]
        self.preposition_map_encoded = {self.word2id[key]: self.word2id[val] for (key, val) in self.preposition_map.items()}
        self.epoch = tf.Variable(0)

    def get_action(self, obs):

        def update_entities(s):
            # Remove unwanted chars
            to_remove = ['\\n', '\\', '|', '$', '/', '>', '_']
            for c in to_remove:
                s = s.replace(c, ' ')

            # Remove multiple whitespaces
            s = re.sub(' +', ' ', s)
            # Remove everything between '-=' and '=-'
            s = re.sub('-=.+?=-', '', s)

            doc = self.nlp(s)
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
                # If unique and not empty, convert to id and add to entity dict
                if noun != '' and noun not in self.entities:
                    n_id = [self.word2id[n] for n in noun.split()]
                    self.entities[noun] = n_id
                if adj_noun != '' and adj_noun not in self.entities:
                    adj_id = [self.word2id[adj] for adj in adj_noun.split()]
                    self.entities[adj_noun] = adj_id
        
        # Update global entity list
        update_entities(obs)
        print(self.entities)

        choices = []
        # For each command
        for cmd in self.possible_cmds_encoded:
            # And each entity
            for _, id1 in self.entities.items():
                # Check if the command has a preposition
                if cmd in self.preposition_map_encoded:
                    preposition = self.preposition_map_encoded[cmd]
                    # And for each entity once again
                    for _, id2 in self.entities.items():
                        # Create an action
                        action = [cmd] + id1 + [preposition] + id2
                        # Make same length
                        action += [self.word2id['<PAD>'] for i in range(cfg['vocab']['max_len'] - len(action))]
                        choices.append(action)
                else:
                    # Create an action
                    action = [cmd] + id1
                    # Make same length
                    action += [self.word2id['<PAD>'] for i in range(cfg['vocab']['max_len'] - len(action))]
                    choices.append(action)
        
        choices = np.array(choices)
        action = self.model(choices)
        #action = choices[np.random.choice(choices.shape[0], 1, replace=False)][0]

        # Convert from ids to words
        final = ''
        for id_ in action:
            if id_ != 0:
                final += self.vocab[id_] + ' '

        return final.strip()
    
    def train(self):
        requested_infos = EnvInfos()

        env_id = textworld.gym.register_games(self.games, requested_infos, max_episode_steps=cfg['train']['max_steps'])
        #env_id = textworld.gym.make_batch(env_id, batch_size=cfg['train']['batch_size'], parallel=False)
        env = gym.make(env_id)

        for epoch in trange(1, cfg['train']['epochs']):
            for game in range(len(self.games)):
                self.entities = {}
                obs, infos = env.reset()

                old_score = 0
                done = False
                while not done:
                    action = self.get_action(obs)
                    print(action)
                    obs, score, done, infos = env.step(action)
                    if score != old_score:
                        print('REWARD!!!')
                        old_score = score

def main():
    model = CustomAgent()
    model.train()

if __name__ == '__main__':
    main()
