# modified from
#
# Natual language Toolkit : BLUE
# Copyright (C) 2001-2013 NLTK Project
# Authors : Chine Yee Lee, Hengteng Li, Ruxin Hou, Calvin Tanujaya Lim
# URL : <http://nltk.org/>
#
#



from __future__ import division

import math, numpy as np
from nltk import word_tokenize
from nltk.compat import Counter
from nltk.util import ngrams

class BLEU(object):
    """
    This class implements the BLEU method, which is used to evaluate
    the quality of machine translation

    reference : Papineni, Koshore. et al. "BLEU: a method for
    automatic evaluation of machine translation", 2002
    """
    
    @staticmethod
    def bleu_score(gen_words, reference, ix_to_word):

        gen_words = [ix_to_word[x] for x in gen_words]
        punctuation = np.argmax(np.array(gen_words) == '.') + 1
        if punctuation == 1:
            punctuation = -1
        gen_words = gen_words[:punctuation]
   
        reference = [ix_to_word[x] for x in reference]
        punctuation = np.argmax(np.array(reference) == '.') + 1
        if punctuation == 1:
            punctutation = -1
        reference = reference[:punctuation]

        gen_sentence = ' '.join(gen_words)
        ref_sentence = ' '.join(reference)

        scores = [BLEU.compute(g, r) for (g, r) in zip(gen_words, reference)]
    
        return np.mean(scores), gen_sentence, ref_sentence

    @staticmethod
    def compute(candidate, reference):

        candidate = [c.lower() for c in candidate]
        reference = [r.lower() for r in reference]

        p_ns = BLEU.modified_precision(candidate, reference, 1)
        bp = BLEU.brevity_penalty(candidate, reference)
        return bp * p_ns

    @staticmethod
    def modified_precision(candidate, reference, n):
        """calculate modified ngram precision"""
        counts = Counter(ngrams(candidate, n))
        if not counts:
            return 0

        max_counts = {}
        
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), \
                    reference_counts[ngram])
        clipped_counts = dict((ngram, min(count, max_counts[ngram])) \
                    for ngram, count in counts.items())
        return sum(clipped_counts.values()) / sum(counts.values())

    @staticmethod
    def brevity_penalty(candidate, reference):
        c = len(candidate)
        r = abs(len(reference)-c)

        if c>r:
            return 1
        else:
            return math.exp(1 - r/c)
