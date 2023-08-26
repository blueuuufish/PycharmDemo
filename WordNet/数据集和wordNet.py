import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')


def get_hypernyms_hyponyms(word):
    hypernyms = set()
    hyponyms = set()

    # 获取单词的所有同义词集
    for syn in wn.synsets(word):
        # 添加上位词到集合
        for hypernym in syn.hypernyms():
            hypernyms.add(hypernym.lemma_names()[0])
        # 添加下位词到集合
        for hyponym in syn.hyponyms():
            hyponyms.add(hyponym.lemma_names()[0])

    return hypernyms, hyponyms


word = "dog"
hypernyms, hyponyms = get_hypernyms_hyponyms(word)

print(f"Hypernyms of {word}: {hypernyms}")
print(f"Hyponyms of {word}: {hyponyms}")



