import spacy


def get_adj_nouns(nlp, entities, vocab):
    doc = nlp(vocab)
    # Get all nouns and adjectives
    for token in doc:
        if token.pos_ is 'NOUN' or token.pos_ is 'ADJ':
            entities.append(token.text)
    
    return entities
'''
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
'''
'''
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
'''
def get_all_cmds(word2id, possible_cmds_encoded, preposition_map_encoded, entities):
    choices = []
    # For each command
    for cmd in possible_cmds_encoded:
        # And each entity
        for _, id1 in entities.items():
            # Check if the command has a preposition
            if cmd in preposition_map_encoded:
                preposition = preposition_map_encoded[cmd]
                # And for each entity once again
                for _, id2 in entities.items():
                    # Create an action
                    action = [cmd] + id1 + [preposition] + id2
                    # Make same length
                    action += [word2id['<PAD>'] for i in range(cfg['vocab']['max_len'] - len(action))]
                    choices.append(action)
            else:
                # Create an action
                action = [cmd] + id1
                # Make same length
                action += [word2id['<PAD>'] for i in range(cfg['vocab']['max_len'] - len(action))]
                choices.append(action)
    
    action = choices[np.random.choice(choices.shape[0], 1, replace=False)][0]
    return action
