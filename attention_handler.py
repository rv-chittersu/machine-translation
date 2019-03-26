import torch.nn as nn
import torch
import torch.nn.functional as f
import math


class Attention(nn.Module):

    def __init__(self, hidden_units, decoder_attention=False, key_value_split=None):
        self.hidden_units = hidden_units
        self.key_value_split = key_value_split
        self.decoder_attention = decoder_attention
        self.intra_attention = False
        self.output_size = hidden_units if key_value_split is None else key_value_split[1]

        self.transform_layer = None
        if key_value_split is not None and hidden_units != sum(key_value_split):
            self.transform_layer = nn.Linear(hidden_units, sum(key_value_split))

        super().__init__()

    def forward(self, current_state, previous_hidden_states, mask):
        if previous_hidden_states is None:
            return None, None
        keys, values = self.split_key_value(previous_hidden_states)
        query = current_state
        attention_weights = self.attention(query, values)
        attention_distribution = self.apply_mask_and_compute_softmax(attention_weights, mask)
        context_vector = self.get_context_vector(attention_distribution, values)
        return context_vector, attention_distribution

    def attention(self, current, previous):
        pass

    def apply_mask_and_compute_softmax(self, weights, mask):
        result = torch.log(mask)
        weights += result
        softmax = f.softmax(weights, dim=0)
        return softmax

    def get_context_vector(self, distribution, previous):
        sequence_len, batch_size, dim = previous.shape
        distribution = distribution.view(sequence_len, batch_size, 1).repeat(1, 1, dim)
        result = torch.mul(distribution, previous)
        result = torch.sum(result, 0)
        return result

    def split_key_value(self, tensor):
        if tensor is None:
            return None, None
        dimensions = len(tensor.shape)
        if self.key_value_split is not None:
            if self.transform_layer is not None:
                tensor = self.transform_layer(tensor)
            return torch.split(tensor, self.key_value_split, dim=dimensions - 1)
        return tensor, tensor


class AdditiveAttention(Attention):

    def __init__(self, hidden_units, decoder_attention=False, key_value_split=None):
        super().__init__(hidden_units, decoder_attention, key_value_split)

        input_size = hidden_units + (hidden_units if key_value_split is None else key_value_split[0])

        self.hidden_layer1 = nn.Linear(input_size, int(input_size/2))
        self.hidden_layer2 = nn.Linear(int(input_size/2), 1)

    def attention(self, current_state, previous_states):
        sequence_length, batch_size, hidden_units = previous_states.shape
        previous_states = previous_states.view(-1, hidden_units)  # (sequence_len*batch_size, hidden_units)
        repeated_current_state = current_state.repeat(sequence_length, 1)  # (sequence_len*batch_size, hidden_units)
        concatenated_vector = torch.cat((previous_states, repeated_current_state), dim=1)
        result = self.hidden_layer1(concatenated_vector)  # (sequence_len*batch_size, hidden_units)
        result = torch.tanh(result)
        result = self.hidden_layer2(result)  # (sequence*batch_len, 1)
        return result.view(sequence_length, batch_size)


class MultiplicativeAttention(Attention):

    def __init__(self, hidden_units, decoder_attention=False, key_value_split=None):
        super().__init__(hidden_units, decoder_attention, key_value_split)

        input_size = hidden_units if key_value_split is None else key_value_split[0]

        self.hidden_layer1 = nn.Linear(input_size, input_size)

    def attention(self, current_state, previous_states):
        sequence_length, batch_size, hidden_units = previous_states.shape
        previous_states = previous_states.view(-1, hidden_units)
        repeated_current_state = current_state.repeat(sequence_length, 1)
        result = self.hidden_layer1(previous_states)
        result = torch.mul(result, repeated_current_state)
        result = torch.sum(result, 1)
        return result.view((sequence_length, batch_size))


class ScaledDotProductAttention(Attention):

    def __init__(self, hidden_units, decoder_attention=False, key_value_split=None):
        super().__init__(hidden_units, decoder_attention, key_value_split)

    def attention(self, current_state, previous_states):
        sequence_length, batch_size, hidden_units = previous_states.shape
        previous_states = previous_states.view(-1, hidden_units)
        repeated_current_state = current_state.repeat(sequence_length, 1)  # (sequence_len*batch_size, hidden_units)
        result = torch.mul(previous_states, repeated_current_state)
        result = torch.sum(result, 1)
        result = torch.div(result, math.sqrt(hidden_units))
        return result.view((sequence_length, batch_size))


class SelfAttention(Attention):

    def __init__(self, hidden_units, decoder_attention=False, key_value_split=None):
        super().__init__(hidden_units, decoder_attention, key_value_split)

        input_size = hidden_units if key_value_split is None else key_value_split[0]

        self.hidden_layer1 = nn.Linear(input_size, int(input_size / 2))
        self.hidden_layer2 = nn.Linear(int(input_size / 2), 1)

    def attention(self, current_state, previous_states):
        sequence_length, batch_size, hidden_units = previous_states.shape
        previous_states = previous_states.view(-1, hidden_units)  # (sequence_len*batch_size, hidden_units)
        result = self.hidden_layer1(previous_states)  # (sequence_len*batch_size, hidden_units)
        result = torch.tanh(result)
        result = self.hidden_layer2(result)  # (sequence*batch_len, 1)
        return result.view(sequence_length, batch_size)
