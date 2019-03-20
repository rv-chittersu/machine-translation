import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as f


class Encoder(nn.Module):

    def __init__(self, vocabulary_size, embedding_size, hidden_units):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_units)

        initial_hidden_state = torch.randn((1, hidden_units), dtype=torch.float)
        self.initial_hidden_state = torch.nn.Parameter(initial_hidden_state, requires_grad=True)

        initial_cell_state = torch.randn((1, hidden_units), dtype=torch.float)
        self.initial_cell_state = torch.nn.Parameter(initial_cell_state, requires_grad=True)

    def forward(self, input_tensor, sequence_lengths):

        sequence_length, batch_size = input_tensor.shape

        hidden_state = self.initial_hidden_state.repeat(batch_size, 1).view(1, batch_size, -1)
        cell_state = self.initial_cell_state.repeat(batch_size, 1).view(1, batch_size, -1)

        embeddings = self.embedding_layer(input_tensor)  # (sequence_length, batch_size, input_dim)

        packed_embeddings = pack_padded_sequence(embeddings, sequence_lengths)

        packed_hidden_states, (hidden_state, cell_state) = self.lstm(packed_embeddings, (hidden_state, cell_state))

        hidden_states = pad_packed_sequence(packed_hidden_states, padding_value=0.0, total_length=sequence_length)

        return hidden_states[0], hidden_state, cell_state


class Decoder(nn.Module):

    hidden_units = 0
    embedding_dimensions = 0

    def __init__(self, vocabulary_size, embedding_size, hidden_units, attention_layer, max_length):
        super().__init__()
        self.hidden_units = hidden_units
        self.embedding_dimensions = embedding_size
        self.max_length = max_length

        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_units)

        if attention_layer is not None:
            self.output_layer = nn.Linear(attention_layer.output_size, vocabulary_size)
        else:
            self.output_layer = nn.Linear(hidden_units, vocabulary_size)

        self.attentionLayer = attention_layer

    def forward(self, output_tensor, encoder_hidden_states, input_mask, hidden_state, cell_state):

        loss = torch.zeros(1)

        _, batch_size = input_mask.shape

        decoder_hidden_states = None
        current_output_mask = None
        result = torch.ones((1, batch_size), dtype=torch.long)

        for position in range(self.max_length - 1):

            embedding_layer_input = result[position] if output_tensor is None else output_tensor[position]
            embeddings = self.embedding_layer(embedding_layer_input)

            _, (hidden_state, cell_state) = self.lstm(embeddings.view(1, batch_size, -1),
                                                      (hidden_state, cell_state))

            if self.attentionLayer is not None:
                output_with_attention, _ = self.attentionLayer(torch.squeeze(hidden_state, 0), encoder_hidden_states,
                                                               input_mask, decoder_hidden_states, current_output_mask)
            else:
                output_with_attention = torch.squeeze(hidden_state, 0)

            dist = self.output_layer(output_with_attention)

            if output_tensor is not None:
                labels = output_tensor[position + 1].clone()
                ce_loss = f.cross_entropy(dist, labels, ignore_index=0, reduction='mean')
                loss += ce_loss

            _, top_indices = dist.data.topk(1)

            if result is None:
                result = top_indices.view(1, -1)
            else:
                result = torch.cat((result, top_indices.view(1, -1)), dim=0)

            if decoder_hidden_states is None:
                decoder_hidden_states = hidden_state.view(1, batch_size, self.hidden_units)
            else:
                decoder_hidden_states = torch.cat((decoder_hidden_states, hidden_state), dim=0)

            current_output_mask = torch.ones((position + 1, batch_size), dtype=torch.float)

        return loss, result
