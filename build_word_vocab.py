import numpy as np

def preProBuildWordVocab(sentence_iterator, word_cnt_threshold=30):
    """
    returns:
        word_to_ix, tx_to_word : words with more than 30 cnts embedding
                ix_to_word[0] = '.', word_to_ix['start'] = 0
    """
    word_cnt = {}
    nsents = 0
    for sent in sentence_iterator : 
        nsents += 1
        for w in sent.lower().split(' '):
            word_cnt[w] = word_cnt.get(w, 0) + 1
    vocab = [w for w in word_cnt if word_cnt[w] >= word_cnt_threshold]
    
    ix_to_word, word_to_ix = {}, {}
    ix_to_word[0] = '.'
    word_to_ix['#START#'] = 0
    ix = 1
    for w in vocab:
        word_to_ix[w] = ix
        ix_to_word[ix] = w
        ix += 1
    word_cnt['.'] = nsents # number of sentence
    bias_init_vector = np.array([1.0*word_cnt[ix_to_word[i]]
        for i in ix_to_word])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)

    return word_to_ix, ix_to_word, bias_init_vector



