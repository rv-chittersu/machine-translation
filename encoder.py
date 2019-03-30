import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from attention_handler import *


class Encoder(nn.Module):

    def __init__(self, vocabulary_size, config):
        super().__init__()
        self.define_modules(vocabulary_size, config.encoder_embedding_size, config.hidden_units,
                            config.layers, config.attention_params)
        self.define_parameters(config.hidden_units, config.layers)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, eps=1e-3, amsgrad=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1, last_epoch=-1)
        self.attention_params = config.attention_params

    def define_modules(self, vocabulary_size, embedding_size, hidden_units,  lstm_layers, attention_params):
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=1)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_units,
                            num_layers=lstm_layers, bidirectional=True)
        self.output_reducer = nn.Linear(2 * hidden_units, hidden_units, bias=False)
        self.hidden_state_reducer = nn.Linear(2 * hidden_units, hidden_units, bias=False)
        self.cell_state_reducer = nn.Linear(2*hidden_units, hidden_units, bias=False)
        self.self_attention = None
        if attention_params["self_attn"]:
            attention_heads = attention_params["heads"]
            print("Encoder: Adding decoder self attention with " + str(attention_heads) + " heads")
            self.self_attention = nn.ModuleList()
            for i in range(attention_heads):
                self.self_attention.append(SelfAttention(2*hidden_units, attention_heads))

    def define_parameters(self, hidden_units, lstm_layers):
        hidden_state = torch.randn((lstm_layers * 2, 1, hidden_units), dtype=torch.float)  # (layers, 1, h_units)
        self.initial_hidden_state = torch.nn.Parameter(hidden_state, requires_grad=True)

        cell_state = torch.randn((lstm_layers * 2, 1, hidden_units), dtype=torch.float)  # (layers, 1, h_units)
        self.initial_cell_state = torch.nn.Parameter(cell_state, requires_grad=True)

    def reset_grad(self):
        self.optimizer.zero_grad()

    def update_weights(self):
        nn.utils.clip_grad_norm_(self.parameters(), 5)
        self.scheduler.step()

    def forward(self, input_tensor, sequence_lengths, input_mask):
        # input_tensor : (sequence_length, batch_size)
        # sequence_lengths : list with len = batch_size

        sequence_length, batch_size = input_tensor.shape

        # scale initial states to batch size
        hidden_state = self.initial_hidden_state.repeat(1, batch_size, 1)  # (layers, batch_size, h_units)
        cell_state = self.initial_cell_state.repeat(1, batch_size, 1)  # (layers, batch_size, h_units)

        # get embeddings
        embeddings = self.embedding_layer(input_tensor)  # (sequence_length, batch_size, input_dim)

        # pack embeddings
        packed_embeddings = pack_padded_sequence(embeddings, sequence_lengths)

        packed_hidden_states, (hidden_state, cell_state) = self.lstm(packed_embeddings, (hidden_state, cell_state))

        # unpack hidden_states
        hidden_states = pad_packed_sequence(packed_hidden_states, padding_value=0.0, total_length=sequence_length)
        hidden_states = hidden_states[0]
        # hidden_states : seq_len, batch, 2*hidden_size
        # hidden_state : 2*layers , batch_size, hidden_size
        # cell_state : layers, batch_size, 2*hidden_size

        _, _, hidden_dimension = hidden_state.shape
        hidden_state = hidden_state.view(-1, 2, batch_size, hidden_dimension)
        cell_state = cell_state.view(-1, 2, batch_size, hidden_dimension)

        hidden_state = torch.cat(torch.split(hidden_state, (1, 1), 1), 3).squeeze(1)
        cell_state = torch.cat(torch.split(cell_state, (1, 1), 1), 3).squeeze(1)

        decoder_cell_state = self.cell_state_reducer(cell_state)
        decoder_hidden_state = self.hidden_state_reducer(hidden_state)

        encoder_attn = None
        if self.self_attention is not None:
            hidden_states = torch.cat(tuple([attention_head(hidden_states, input_mask) for attention_head in self.self_attention]), dim=2)

        output = self.hidden_state_reducer(hidden_states)

        return output, decoder_hidden_state, decoder_cell_state, encoder_attn
