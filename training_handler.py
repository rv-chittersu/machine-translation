from encoder import Encoder
from decoder import Decoder


class Trainer:

    def __init__(self, encoder, decoder):
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def feed_mini_batch(self, input_tensor, input_lengths, input_mask,  output_tensor, output_lengths,
                        output_mask, train=True, result_file=None, attention_file=None):
        if train:
            self.encoder.reset_grad()
            self.decoder.reset_grad()

        hidden_states, hidden_state, cell_state = self.encoder(input_tensor, input_lengths)
        loss, result, attn = self.decoder(output_tensor, hidden_states, input_mask, hidden_state, cell_state)

        if train:
            loss.backward()
            self.encoder.update_weights()
            self.decoder.update_weights()

        # if result_file is not None:
        #     # write the translated sentence to file
        #     pass
        #
        # if attention_file is not None:
        #     # write attention to file
        #     pass

    def train(self, source_language_file, destination_language_file):
        # from file read a mini batch
        #
        pass
