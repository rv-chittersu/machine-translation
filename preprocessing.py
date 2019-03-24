import datetime
import pandas as pd
from torchtext.data import Field, TabularDataset, BucketIterator
from sklearn.model_selection import train_test_split
import nltk
from config_handler import Config
from utils import *


def remove_punctuation(string):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ch in punctuations:
            if ch in string:
                string = string.replace(ch, '')
        return string


def english_tokenizer(sentence):
    result = []
    for c in contractions.keys():
        if c in sentence:
            res = sentence.split(c)
            sentence = contractions[c].split("/")[0].strip().join(res)
    for sent in nltk.sent_tokenize(sentence, language='english'):
        result = result + (nltk.word_tokenize(sent, language='english'))
    return result


def german_tokenizer(sentence):
    result = []
    for sent in nltk.sent_tokenize(sentence, language='german'):
        result = result + (nltk.word_tokenize(sent, language='german'))
    return result


def hindi_tokenizer(sentence):
    sentence = remove_punctuation(sentence)
    return sentence.split()


def pre_process(tokenized_sentence):
    result = []
    for word in tokenized_sentence:
        word = str.strip(word)
        word = remove_punctuation(word)
        if str.isdigit(word):
            word = '<num>'
        if len(word) != 0:
            result.append(word)
    if len(result) > 60:
        return []
    return result


def get_field(language):
    if language == 'en':
        return Field(tokenize=english_tokenizer, preprocessing=pre_process,
                     lower=True, init_token="<sos>", eos_token="<eos>")
    elif language == 'de':
        return Field(tokenize=german_tokenizer, preprocessing=pre_process,
                     lower=True, init_token="<sos>", eos_token="<eos>")
    elif language == 'hi':
        return Field(tokenize=hindi_tokenizer, preprocessing=pre_process,
                     lower=True, init_token="<sos>", eos_token="<eos>")


def initialize_lang_fields(file, language1, language2, freq):

    lang1 = get_field(language1)
    lang2 = get_field(language2)

    train_data = read(file, lang1, lang2)

    lang1.build_vocab(train_data, min_freq=freq)
    lang2.build_vocab(train_data, min_freq=freq)

    return lang1, lang2, train_data


def save_vocab(lang, file):
    f = open(file, 'w')
    mapping = lang.vocab.stoi

    for key in mapping.keys():
        f.write(key + "," + str(mapping[key]) + '\n')
    f.close()


def limit_len(file1):
    with open(file1) as f:
        lines = f.readlines()

    f = open(file1, 'w')
    count = 0
    for line in lines:
        pair = line.split("\t")
        if len(pair) != 2:
            continue
        if max(len(pair[1].split(" ")), len(pair[0].split(" "))) > config.max_sent_length:
            count += 1
            continue
        f.write(line)
    f.close()
    print("skipped " + str(count) + " lines")


def write_to_file(data, file):

    count = 0
    f = open(file, 'w')
    data_size = len(data.examples)
    iterator = BucketIterator(dataset=data, batch_size=200, repeat=False)

    for i in range(data_size):
        b = next(iter(iterator))
        results = process_encoded_sentences(b.lang1, b.lang2)
        f.write("\n".join(results))
        count += 200

        if count > data_size:
            break

        if count % 10000 == 0:
            print(str(datetime.datetime.now()) + ": " + file + ":" + str(count) + "/" + str(data_size))
    f.close()


def read(file, lang1, lang2):
    data_fields = [('lang1', lang1), ('lang2', lang2)]
    res = TabularDataset(path=file, format='tsv', fields=data_fields, skip_header=True)
    print(str(datetime.datetime.now()) + ": loaded " + file)
    return res


if __name__ == '__main__':

    config = Config('./config.ini')

    english_file = open(config.source_data, encoding='utf-8').read().split('\n')
    german_file = open(config.destination_data, encoding='utf-8').read().split('\n')

    raw_data = {'lang1': [line.replace('\t', ' ') for line in english_file],
                'lang2': [line.replace('\t', ' ') for line in german_file]}

    df = pd.DataFrame(raw_data, columns=["lang1", "lang2"])

    df['lang1_len'] = df['lang1'].str.count(' ')
    df['lang2_len'] = df['lang2'].str.count(' ')

    l = df.index[(df['lang1'] == 0) | (df['lang2'] == 0)].tolist()

    s = set()

    l = map(lambda index: s.add([index-1, index, index+1]), l)

    df = df.drop(df.index[list(s)])

    df = df.query('lang1_len < ' + str(config.max_sent_length) + ' & lang2_len < ' + str(config.max_sent_length) +  ' & lang1_len > 5 & lang2_len > 5')
    df = df.query('lang2_len < lang1_len * 1.5 & lang2_len * 1.5 > lang1_len')

    train, test = train_test_split(df, test_size=0.2)
    train, dev = train_test_split(train, test_size=0.2)

    print(str(datetime.datetime.now()) + ": creating processed files")
    # write train test dev to file
    train.to_csv(config.training_data, index=False, sep='\t')
    dev.to_csv(config.dev_data, index=False, sep='\t')
    test.to_csv(config.test_data, index=False, sep='\t')

    print(str(datetime.datetime.now()) + ": creating vocab from training data")
    lang1, lang2, train_data = initialize_lang_fields(config.training_data, config.source_lang, config.destination_lang, config.min_freq)

    print(str(datetime.datetime.now()) + ": saving vocab")
    save_vocab(lang1, config.source_vocab)
    save_vocab(lang2, config.destination_vocab)

    print(str(datetime.datetime.now()) + ": saving training data")
    write_to_file(train_data, config.processed_training_data)
    limit_len(config.processed_training_data)

    dev = read(config.dev_data, lang1, lang2)
    print(str(datetime.datetime.now()) + ": saving dev data")
    write_to_file(dev, config.processed_dev_data)
    limit_len(config.processed_dev_data)

    test = read(config.test_data, lang1, lang2)
    print(str(datetime.datetime.now()) + ": saving test data")
    write_to_file(test, config.processed_test_data)
    limit_len(config.processed_test_data)

