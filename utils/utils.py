import spacy


def get_adj_nouns(nlp, entities, vocab):
    doc = nlp(vocab)
    # Get all nouns and adjectives
    for token in doc:
        if token.pos_ is 'NOUN' or token.pos_ is 'ADJ':
            entities.append(token.text)
    
    return entities

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
