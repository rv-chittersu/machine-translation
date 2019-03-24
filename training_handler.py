from encoder import Encoder
from decoder import Decoder
from torchtext.data import BucketIterator, TabularDataset, Field
from torchnlp.metrics import get_moses_multi_bleu
from utils import *
import numpy as np
from datetime import datetime as dt


def post(batch, vocab):
    return np.array(list(batch), dtype=np.long)


class Trainer:

    def __init__(self, encoder, decoder, file_name):

        self.lang1 = Field(use_vocab=False, postprocessing=post, pad_token='1')
        self.lang2 = Field(use_vocab=False, postprocessing=post, pad_token='1')
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.key = file_name

    def clean_grads(self):
        for p in self.encoder.parameters():
            if p.grad is not None:
                del p.grad  # free some memory
        for p in self.decoder.parameters():
            if p.grad is not None:
                del p.grad  # free some memory

    def feed_mini_batch(self, input_tensor,  output_tensor, mode):
        # operation such that all non zero is one and all one is zero
        input_mask = torch.FloatTensor(input_tensor.shape).copy_(input_tensor).apply_(lambda val: 0 if val == 1 else 1).cuda()
        input_lengths = torch.sum(input_mask, 0)

        if mode == 'train':
            self.encoder.reset_grad()
            self.decoder.reset_grad()

        hidden_states, hidden_state, cell_state = self.encoder(input_tensor, input_lengths)

        if mode == 'train' or mode == 'dev':
            loss, result, _ = self.decoder(output_tensor, hidden_states, input_mask, hidden_state, cell_state, 0)
        else:
            loss, result, _ = self.decoder(None, hidden_states, input_mask, hidden_state, cell_state, 60)

        if mode == 'train':
            self.encoder.update_weights()
            self.decoder.update_weights()

        loss = float(loss)

        score = 0
        if mode == 'test':
            result1, result2 = process_encoded_sentences(result, output_tensor, sep=True)
            score = get_moses_multi_bleu(result1, result2)
            score = 0 if score is None else score

        torch.cuda.empty_cache()
        return loss, score

    def run(self, file, batch_size, max_batches, mode):

        data_fields = [('lang1', self.lang1), ('lang2', self.lang2)]
        data = TabularDataset(path=file, format='tsv', fields=data_fields, skip_header=True)

        batch_iterator = BucketIterator(data, batch_size=batch_size, sort_key=lambda x: len(x.lang1),
                                        shuffle=True, sort_within_batch=True, repeat=False)

        data_count = len(data.examples)
        print(str(dt.now()) + ": " + mode + " data size - " + str(data_count))

        total_loss = 0
        total_score = 0
        batches = 0

        intermediate_loss = 0
        intermediate_score = 0
        intermediate_batches = 0

        while True:
            batch = next(iter(batch_iterator))
            loss, score = self.feed_mini_batch(batch.lang1.cuda(), batch.lang2.cuda(), mode)
            total_loss += loss
            total_score += score
            intermediate_loss += loss
            intermediate_score += score
            intermediate_batches += 1
            batches += 1

            if batches % 100 == 0:
                if mode != 'test':
                    print(str(dt.now()) + ": " + mode + "@" + str(batches) + " loss:" + str(intermediate_loss/intermediate_batches))
                else:
                    print(str(dt.now()) + ": " + mode + "@" + str(batches) + " score:" + str(intermediate_score/intermediate_batches))
                intermediate_loss = 0
                intermediate_batches = 0
                intermediate_score = 0

            if batches*batch_size >= data_count:
                break
            if batches == max_batches:
                break
        return total_loss, total_score, batches
