from torch import optim
from attention_handler import *


class Decoder(nn.Module):

    max_length = 0

    lstm = None
    embedding_layer = None
    attention_layer = None
    output_layer = None
    optimizer = None

    def __init__(self, vocabulary_size, embedding_size, hidden_units, attention_params, max_length, learning_rate):
        super().__init__()
        self.max_length = max_length
        self.define_layers(vocabulary_size, embedding_size, hidden_units, attention_params)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def define_layers(self, vocabulary_size, embedding_size, hidden_units, layers, attention_params=None):
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_units, num_layers=layers)
        self.attention_layer = self.define_attention_layer(attention_params, hidden_units)

        output_layer_size = hidden_units
        if self.attention_layer is not None:
            output_layer_size = self.attention_layer.output_size
        self.output_layer = nn.Linear(output_layer_size, vocabulary_size)

    def define_attention_layer(self, params, hidden_units):
        if params is None:
            return
        if params.name == 'additive':
            return AdditiveAttention(hidden_units, params.self_attn, params.key_value_split)
        elif params.name == 'multiplicative':
            return MultiplicativeAttention(hidden_units, params.self_attn, params.key_value_split)
        elif params.name == 'scaled_dot_product':
            return ScaledDotProductAttention(hidden_units, params.self_attn, params.key_value_split)
        return

    def reset_grad(self):
        self.optimizer.zero_grad()

    def update_weights(self):
        self.optimizer.step()

    def forward(self, output_tensor, encoder_hidden_states, input_mask, hidden_state, cell_state):
        # define loss
        loss = torch.zeros(1)

        layers, batch_size, hidden_size = hidden_state.shape

        # decoder hidden_states tensor
        decoder_hidden_states = None

        # variable for output mask
        current_output_mask = None

        # result generated by decoder
        result = torch.ones((1, batch_size), dtype=torch.long)  # (1, batch_size)

        attn = []

        for position in range(self.max_length - 1):

            # if output is absent get input from previous step result and generate embeddings
            embedding_layer_input = result[position] if output_tensor is None else output_tensor[position]
            embeddings = self.embedding_layer(embedding_layer_input)  # (batch_size, embedding_dim)

            lstm_output, (hidden_state, cell_state) = self.lstm(embeddings.view(1, batch_size, -1),
                                                                       (hidden_state, cell_state))

            # final_hidden_state : 1, batch_size, hidden_units
            lstm_output = lstm_output.view(batch_size, -1)  # (batch_size, hidden_units)

            # pass final_hidden_state through attention layer exist
            if self.attention_layer is not None:
                output_with_attention, attn_dist = self.attention_layer(lstm_output, encoder_hidden_states,
                                                                input_mask, decoder_hidden_states, current_output_mask)
            else:
                output_with_attention = lstm_output
                attn_dist = None

            # get dist on vocabulary
            dist = self.output_layer(output_with_attention)

            # calculate loss if output is available
            if output_tensor is not None:
                ce_loss = f.cross_entropy(dist, output_tensor[position + 1], ignore_index=0, reduction='mean')
                loss += ce_loss

            # get top predictions
            _, top_indices = dist.data.topk(1)

            if result is None:
                result = top_indices.view(1, -1)
            else:
                result = torch.cat((result, top_indices.view(1, -1)), dim=0)

            if decoder_hidden_states is None:
                decoder_hidden_states = lstm_output.view(1, batch_size, -1)
            else:
                decoder_hidden_states = torch.cat((decoder_hidden_states, lstm_output), dim=0)

            if attn_dist is not None:
                attn.append(attn_dist)

            current_output_mask = torch.ones((position + 1, batch_size), dtype=torch.float)

        return loss, result, attn