from torch import optim
from attention_handler import *


class Decoder(nn.Module):

    def __init__(self, vocabulary_size, config):
        super().__init__()
        # embedding_size, hidden_units, attention_params, max_length, learning_rate
        self.define_layers(vocabulary_size, config.decoder_embedding_size, config.hidden_units,
                           config.layers, config.attention_params)
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate, eps=1e-3, amsgrad=True)
        self.attention_params = config.attention_params

    def define_layers(self, vocabulary_size, embedding_size, hidden_units, layers, attention_params=None):
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_units, num_layers=layers)
        self.define_attention_layer(attention_params, hidden_units)
        output_layer_size = hidden_units
        if self.attention_layer is not None:
            output_layer_size += self.attention_layer.output_size
        if self.self_attention_layer is not None:
            output_layer_size += 2*self.self_attention_layer.output_size
        self.output_layer = nn.Linear(output_layer_size, vocabulary_size)

    def define_attention_layer(self, params, hidden_units):
        name = params['name']
        self_attn = params['self_attn']
        key_value_split = params['key_value_split']
        if params is None:
            self.attention_layer = None
            self.self_attention_layer = None
            return
        if name == 'additive':
            self.attention_layer = AdditiveAttention(hidden_units, key_value_split)
        elif name == 'multiplicative':
            self.attention_layer = MultiplicativeAttention(hidden_units, key_value_split)
        elif name == 'scaled_dot_product':
            self.attention_layer = ScaledDotProductAttention(hidden_units, key_value_split)
        else:
            self.attention_layer = None

        if self_attn:
            self.self_attention_layer = SelfAttention(hidden_units, key_value_split)
        else:
            self.self_attention_layer = None
        return

    def reset_grad(self):
        self.optimizer.zero_grad()

    def update_weights(self):
        self.optimizer.step()

    def get_output_with_attention(self, value, attn, decoder_self_attn, encoder_self_attn):
        batch_size, _ = value.shape
        if encoder_self_attn is not None and decoder_self_attn is None:
            # probably first stage in self decoder attn
            # expected attn layer output size
            decoder_self_attn = torch.zeros((batch_size, self.self_attention_layer.output_size), device=torch.device('cuda'))
        outputs = [value]
        if attn is not None:
            outputs.append(attn)
        if decoder_self_attn is not None:
            outputs.append(decoder_self_attn)
            outputs.append(encoder_self_attn)
        return torch.cat(outputs, dim=1)

    def forward(self, output_tensor, encoder_hidden_states, input_mask, hidden_state, cell_state, encoder_attention, max_length):
        # define loss
        loss = 0

        layers, batch_size, hidden_size = hidden_state.shape

        seq_len = max_length if output_tensor is None else output_tensor.shape[0]

        # decoder hidden_states tensor
        decoder_hidden_states = None

        # variable for output mask
        current_output_mask = None

        # result generated by decoder
        result = torch.empty((1, batch_size), dtype=torch.long, device=torch.device('cuda'))  # (1, batch_size)
        result.fill_(2)

        if self.attention_params['name'] == 'self_attention':
            encoder_hidden_states = None
            input_mask = None

        for position in range(seq_len - 1):

            # if output is absent get input from previous step result and generate embeddings
            embedding_layer_input = result[position] if output_tensor is None else output_tensor[position]
            embeddings = self.embedding_layer(embedding_layer_input)  # (batch_size, embedding_dim)
            lstm_output, (hidden_state, cell_state) = self.lstm(embeddings.view(1, batch_size, -1),
                                                                       (hidden_state, cell_state))

            # final_hidden_state : 1, batch_size, hidden_units
            lstm_output = lstm_output.view(batch_size, -1)  # (batch_size, hidden_units)

            # pass final_hidden_state through attention layer exist
            context = None
            self_context = None
            if self.attention_layer is not None:
                val, context, _ = self.attention_layer(lstm_output, encoder_hidden_states, input_mask)
            if self.self_attention_layer is not None:
                val, self_context, _ = self.self_attention_layer(None, decoder_hidden_states, current_output_mask)
            output_with_attention = self.get_output_with_attention(lstm_output, context, self_context, encoder_attention)

            # get dist on vocabulary
            dist = self.output_layer(output_with_attention)

            # calculate loss if output is available
            if output_tensor is not None:
                ce_loss = f.cross_entropy(dist, output_tensor[position + 1], ignore_index=1, reduction='mean')
                ce_loss.backward(retain_graph=True)
                loss += float(ce_loss)

            # get top predictions
            _, top_indices = dist.data.topk(1)

            if result is None:
                result = top_indices.view(1, -1)
            else:
                result = torch.cat((result, top_indices.view(1, -1)), dim=0)

            if decoder_hidden_states is None:
                decoder_hidden_states = lstm_output.view(1, batch_size, -1)
            else:
                decoder_hidden_states = torch.cat((decoder_hidden_states, lstm_output.view(1, batch_size, -1)), dim=0)

            current_output_mask = torch.ones((position + 1, batch_size), dtype=torch.float, device=torch.device('cuda'))

        return loss/seq_len, result
