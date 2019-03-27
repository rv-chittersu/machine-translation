import torch
from nltk.translate.bleu_score import *


contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}


def process_encoded_tensor(t):
    result = []
    s = torch.transpose(t, 0, 1).tolist()

    for sen in s:
        l = len(sen)
        if 3 in sen:
            l = sen.index(3) + 1
        result.append(" ".join([str(k) for k in sen[0: l]]))
    return result


def process_encoded_sentences(t1, t2, sep=False):
    result1 = []
    result2 = []

    s1 = torch.transpose(t1, 0, 1).tolist()
    s2 = torch.transpose(t2, 0, 1).tolist()

    for j in range(len(s1)):

        len1 = len(s1[j])
        len2 = len(s2[j])

        if 3 in s1[j]:
            len1 = s1[j].index(3) + 1
        if 3 in s2[j]:
            len2 = s2[j].index(3) + 1
        str1 = " ".join([str(k) for k in s1[j][0: len1]])
        str2 = " ".join([str(k) for k in s2[j][0: len2]])

        if sep:
            result1.append(str1)
            result2.append(str2)
        else:
            result1.append(str1 + '\t' + str2)

    return [result1, result2] if sep else result1


def write_to_file(file, list1, list2=None):
    f = open(file, "w+")
    if list2 is None:
        f.write("\n".join(list1) + "\n")
    else:
        for index, sen1 in enumerate(list1):
            sen2 = list2[index]
            f.write(sen1 + '\t' + sen2 + '\n')
    f.close()


def get_vocab_size(file):
    size = 0
    with open(file, 'r') as f:
        lines = f.read().split("\n")
        for line in lines:
            if len(line.split(",")) != 2:
                continue
            size += 1
    return size


def compute_bleu_score(file):
    hypothesis = [x.split(" ") for x in open(file + ".hyp").read().split("\n") if len(x) != 0]
    references = [[x.split(" ")] for x in open(file + ".ref").read().split("\n") if len(x) != 0]
    return corpus_bleu(references, hypothesis)


def get_reverse_vocab(file):
    with open(file, 'r') as f:
        lines = f.read().split("\n")
        result = [x.split(",")[0] for x in lines if len(x) != 0]
        print(result)
    return result


def decode(encoded_sentence, revers_vocab):
    result = []
    for word in encoded_sentence.split(" "):
        act_word = revers_vocab[int(word)]
        result.append(act_word)
    return " ".join(result)
