import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from attention_handler import *


class Encoder(nn.Module):
    embedding_layer = None
    lstm = None
    initial_hidden_state = None
    initial_cell_state = None
    optimizer = None

    output_reducer = None
    hidden_state_reducer = None
    cell_state_reducer = None

    def __init__(self, vocabulary_size, config):
        super().__init__()
        # embedding_size, hidden_units, lstm_layers, learning_rate
        self.define_modules(vocabulary_size, config.embedding_size, config.hidden_units,  config.lstm_layers)
        self.define_parameters(config.hidden_units, config.lstm_layers)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)

    def define_modules(self, vocabulary_size, embedding_size, hidden_units,  lstm_layers):
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=1)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_units,
                            num_layers=lstm_layers, bidirectional=True)
        self.output_reducer = nn.Linear(2*hidden_units, hidden_units, bias=False)
        self.hidden_state_reducer = nn.Linear(2*hidden_units, hidden_units, bias=False)
        self.cell_state_reducer = nn.Linear(2*hidden_units, hidden_units, bias=False)

    def define_parameters(self, hidden_units, lstm_layers):
        initial_hidden_state = torch.randn((lstm_layers, 1, hidden_units), dtype=torch.float)  # (layers, 1, h_units)
        self.initial_hidden_state = torch.nn.Parameter(initial_hidden_state, requires_grad=True)

        initial_cell_state = torch.randn((lstm_layers, 1, hidden_units), dtype=torch.float)  # (layers, 1, h_units)
        self.initial_cell_state = torch.nn.Parameter(initial_cell_state, requires_grad=True)

    def reset_grad(self):
        self.optimizer.zero_grad()

    def update_weights(self):
        self.optimizer.step()

    def forward(self, input_tensor, sequence_lengths):
        # input_tensor : (sequence_length, batch_size)
        # sequence_lengths : list with len = batch_size

        sequence_length, batch_size = input_tensor.shape

        # scale initial states to batch size
        hidden_state = self.initial_hidden_state.repeat(1, batch_size, 1)  # (layers, batch_size, h_units)
        cell_state = self.initial_cell_state.repeat(batch_size, 1)  # (layers, batch_size, h_units)

        # get embeddings
        embeddings = self.embedding_layer(input_tensor)  # (sequence_length, batch_size, input_dim)

        # pack embeddings
        packed_embeddings = pack_padded_sequence(embeddings, sequence_lengths)

        packed_hidden_states, (hidden_state, cell_state) = self.lstm(packed_embeddings, (hidden_state, cell_state))

        # unpack hidden_states
        hidden_states = pad_packed_sequence(packed_hidden_states, padding_value=0.0, total_length=sequence_length)

        # hidden_states : seq_len, batch, 2*hidden_size
        # hidden_state : layers , batch_size, 2*hidden_size
        # cell_state : layers, batch_size, 2*hidden_size

        output = self.output_reducer(hidden_states)
        decoder_hidden_state = self.hidden_state_reducer(hidden_state)
        decoder_cell_state = self.cell_state_reducer(cell_state)

        return output, decoder_hidden_state, decoder_cell_state
