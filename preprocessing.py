import datetime
import pandas as pd
from torchtext.data import Field, TabularDataset
from sklearn.model_selection import train_test_split
import spacy
from config_handler import Config


def remove_punctuation(string):
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for ch in punctuations:
            if ch in string:
                string = string.replace(ch, '')
        return string


def english_tokenizer(sentence):
    en = spacy.load('en')
    return [tok.text for tok in en.tokenizer(sentence)]


def german_tokenizer(sentence):
    de = spacy.load('de')
    return [tok.text for tok in de.tokenizer(sentence)]


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


def initialize_lang_fields(file, language1, language2):

    lang1 = get_field(language1)
    lang2 = get_field(language2)

    data_fields = [('lang1', lang1), ('lang2', lang2)]
    train_data = TabularDataset(path=file, format='tsv', fields=data_fields, skip_header=True)

    print(datetime.datetime.now())
    lang1.build_vocab(train_data, min_freq=100)
    print(datetime.datetime.now())
    lang2.build_vocab(train_data, min_freq=100)
    print(datetime.datetime.now())

    return lang1, lang2, train_data


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

    df = df.query('lang1_len < 80 & lang2_len < 80 & lang1_len > 5 & lang2_len > 5')
    df = df.query('lang2_len < lang1_len * 1.5 & lang2_len * 1.5 > lang1_len')

    train, test = train_test_split(df, test_size=0.2)
    train, dev = train_test_split(train, test_size=0.2)

    # write train test dev to file
    train.to_csv(config.processed_training_data, index=False, sep='\t', header=False)
    dev.to_csv(config.processed_dev_data, index=False, sep='\t', header=False)
    test.to_csv(config.processed_test_data, index=False, sep='\t', header=False)

    # lang1, lang2, train_data = initialize_lang_fields(config.processed_training_data, config.source_lang, config.destination_lang)
