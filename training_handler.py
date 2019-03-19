from torch import optim


class Trainer:

    def __init__(self, encoder, decoder, attention_layer,  max_input_length, max_output_length):
        self.encoder = encoder
        self.decoder = decoder
        self.attention_layer = attention_layer
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
        self.decoder_optimizer = optim.Adam(decoder.paramters(), lr=0.01)
        self.attention_layer_optimizer = optim.Adam(attention_layer.parameters(), lr=0.01)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def feed_mini_batch(self, input_tensor, input_lengths, input_mask,  output_tensor, output_mask, train=True, result_file=None):
        if train:
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            self.attention_layer_optimizer.zero_grad()

        hidden_states, hidden_state, cell_state = self.encoder(input_tensor, input_lengths)
        loss, result = self.decoder(output_tensor, output_mask, hidden_states, input_mask, hidden_state, cell_state)

        if train:
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            self.attention_layer_optimizer.step()

        if result_file is not None:
            # write the translated sentence to file
            pass

    def train(self, source_language_file, destination_language_file):
        # from file read a mini batch
        #
        pass
