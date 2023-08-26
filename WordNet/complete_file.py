import json
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_hypernyms_hyponyms(word):
    hypernyms = set()
    hyponyms = set()

    for syn in wn.synsets(word):
        for hypernym in syn.hypernyms():
            hypernyms.add(hypernym.lemma_names()[0])
        for hyponym in syn.hyponyms():
            hyponyms.add(hyponym.lemma_names()[0])

    return list(hypernyms), list(hyponyms)


def replace_with_hypernym_or_hyponym(word, pos):
    # Map POS tag to first character used by WordNet
    tag = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}.get(pos[0], None)
    if tag:
        hypernyms, hyponyms = get_hypernyms_hyponyms(word)
        # Return the first hypernym or hyponym if available, else return the original word
        return hypernyms[0] if hypernyms else (hyponyms[0] if hyponyms else word)
    return word


with open('train.json', 'r') as file:
    data = json.load(file)

    if isinstance(data, list):
        for item in data[:4]:
            words = item['words']
            pos_tags = nltk.pos_tag(words)
            replaced_words = [replace_with_hypernym_or_hyponym(word, pos) for word, pos in pos_tags]

            # Print original and replaced sentences
            print("Original: ", ' '.join(words))
            print("Replaced: ", ' '.join(replaced_words))
            print("-------------------------------------------------")
