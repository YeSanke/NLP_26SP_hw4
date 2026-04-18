import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    qwerty_neighbors = {
        'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'srfce', 'e': 'wsdr',
        'f': 'dgrtc', 'g': 'fhjty', 'h': 'gjkyu', 'i': 'ujko', 'j': 'hkniu',
        'k': 'jlmo', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }
 
    def get_synonym(word):
        synsets = wordnet.synsets(word)
        lemmas = []
        for syn in synsets:
            for lemma in syn.lemmas():
                candidate = lemma.name().replace('_', ' ')
                if candidate.lower() != word.lower():
                    lemmas.append(candidate)
        return random.choice(lemmas) if lemmas else None
 
    def introduce_typo(word):
        # swap two adjacent characters in the word
        if len(word) < 2:
            return word
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
 
    words = word_tokenize(example["text"])
    new_words = []
    for word in words:
        r = random.random()
        if r < 0.3:      # 30% chance: try synonym replacement
            synonym = get_synonym(word)
            new_words.append(synonym if synonym else word)
        elif r < 0.45:   # 15% chance: introduce typo
            new_words.append(introduce_typo(word))
        else:
            new_words.append(word)
 
    example["text"] = TreebankWordDetokenizer().detokenize(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
