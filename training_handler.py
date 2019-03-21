from encoder import Encoder
from decoder import Decoder
from torchtext.data import Field, BucketIterator, TabularDataset
import torch


class Trainer:

    def __init__(self, lang1, lang2, encoder, decoder, mode, key):
        self.lang1 = lang1
        self.lang2 = lang2
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.mode = mode
        self.key = key
        self.result_file_name = key+mode + '.res'

    def feed_mini_batch(self, input_tensor,  output_tensor):
        # operation such that all non zero is one and all one is zero
        input_mask = input_tensor.clone().apply_(lambda val: 0 if val == 1 else 1)
        input_lengths = torch.sum(input_mask, 0)

        if self.mode == 'train':
            self.encoder.reset_grad()
            self.decoder.reset_grad()

        hidden_states, hidden_state, cell_state = self.encoder(input_tensor, input_lengths)
        loss, result, _ = self.decoder(output_tensor, hidden_states, input_mask, hidden_state, cell_state)

        if self.mode == 'train':
            loss.backward()
            self.encoder.update_weights()
            self.decoder.update_weights()

        if self.mode == 'eval' or self.mode == 'test':
            # write translated stuff to file
            EN_TEXT.vocab.itos[11]
            pass

    def run(self, file, batch_size, batches):

        data_fields = [('lang1', self.lang1), ('lang2', self.lang2)]
        data = TabularDataset(path='./train.en.de', train=file, format='tsv', fields=data_fields, skip_header=True)

        batch_iterator = BucketIterator(data, batch_size=batch_size, sort_key=lambda x: len(x.lang1),
                                        shuffle=True, sort_within_batch=True, repeat=False)

        for i in range(batches):
            try:
                batch = next(iter(batch_iterator))
                self.feed_mini_batch(batch.lang1, batch.lang2)
            except StopIteration:
                break
