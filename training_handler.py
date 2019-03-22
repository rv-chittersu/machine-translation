from encoder import Encoder
from decoder import Decoder
from torchtext.data import BucketIterator, TabularDataset
from torchnlp.metrics import get_moses_multi_bleu
from utils import *


class Trainer:

    def __init__(self, lang1, lang2, encoder, decoder, key):
        self.lang1 = lang1
        self.lang2 = lang2
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.key = key

    def get_result_file_name(self, mode):
        return self.key + '.' + mode + '.res'

    def feed_mini_batch(self, input_tensor,  output_tensor, mode):
        # operation such that all non zero is one and all one is zero
        input_mask = input_tensor.clone().apply_(lambda val: 0 if val == 1 else 1)
        input_lengths = torch.sum(input_mask, 0)

        if mode == 'train':
            self.encoder.reset_grad()
            self.decoder.reset_grad()

        hidden_states, hidden_state, cell_state = self.encoder(input_tensor, input_lengths)

        if mode == 'train':
            loss, result, _ = self.decoder(output_tensor, hidden_states, input_mask, hidden_state, cell_state, 0)
        else:
            loss, result, _ = self.decoder(None, hidden_states, input_mask, hidden_state, cell_state, 1.5*output_tensor.shape[0])

        if mode == 'train':
            loss.backward()
            self.encoder.update_weights()
            self.decoder.update_weights()

        score = 0
        if mode == 'test' or mode == 'dev':
            result1, result2 = process_encoded_sentences(result, self.lang2, sep=True)
            score = get_moses_multi_bleu(result1, result2)

            if mode == 'test':
                write_to_file(self.get_result_file_name(mode), result1, result2)

        return loss, score

    def run(self, file, batch_size, mode):

        data_fields = [('lang1', self.lang1), ('lang2', self.lang2)]
        data = TabularDataset(path='./train.en.de', train=file, format='tsv', fields=data_fields, skip_header=True)

        batch_iterator = BucketIterator(data, batch_size=batch_size, sort_key=lambda x: len(x.lang1),
                                        shuffle=True, sort_within_batch=True, repeat=False)

        data_count = data.examples
        print(mode + " data size - " + len(data.examples))

        total_loss = 0
        total_score = 0
        batches = 0

        intermediate_loss = 0

        while True:
            batch = next(iter(batch_iterator))
            loss, score = self.feed_mini_batch(batch.lang1, batch.lang2, mode)
            total_loss += loss
            intermediate_loss += loss
            total_score += score
            batches += 1

            if batches % 100 == 0:
                print(mode + "@" + str(batches) + " loss:" + str(intermediate_loss/batches))
                intermediate_loss = 0

            if batches*batch_size >= data_count:
                break
