# Define model

from torch.nn import Module
import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import Embedding
from torch.nn import MultiheadAttention
from torch.nn import AvgPool1d
from torch.nn import LayerNorm
from torch.nn import Conv1d
from torch.nn import Dropout
from torch.nn import Sequential
from torch.nn import Linear
import torch.nn.init as init
import math
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MHA, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query_linear = nn.Linear(input_size, hidden_size * num_heads)
        self.key_linear = nn.Linear(input_size, hidden_size * num_heads)
        self.value_linear = nn.Linear(input_size, hidden_size * num_heads)

        self.output_linear = nn.Linear(hidden_size * num_heads, input_size)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()

        # Apply linear transformations to obtain queries, keys, and values
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)

        # Reshape queries, keys, and values into multiple heads
        queries = queries.view(
            batch_size, seq_length, self.num_heads, self.hidden_size
        ).transpose(1, 2)
        keys = keys.view(
            batch_size, seq_length, self.num_heads, self.hidden_size
        ).transpose(1, 2)
        values = values.view(
            batch_size, seq_length, self.num_heads, self.hidden_size
        ).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (self.hidden_size**0.5)

        # Apply mask, if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(
                2
            )  # Expand mask dimensions for broadcasting
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_probs, values)

        # Reshape and concatenate attention outputs
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(
            batch_size, seq_length, self.hidden_size * self.num_heads
        )

        # Apply linear transformation to obtain the final output
        output = self.output_linear(attention_output)

        return output, torch.mean(attention_probs, dim=1)


class Encoder(Module):
    def __init__(
        self, d_model: int, d_hidden: int, q: int, v: int, h: int, dropout: float = 0.1
    ):
        super(Encoder, self).__init__()

        self.mha = MHA(d_model, q, h)

        self.conv1d = Sequential(
            nn.Conv1d(d_model, d_model, 1), nn.ReLU(), nn.Conv1d(d_model, d_model, 1)
        )

        self.dropout = torch.nn.Dropout(p=dropout)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, mask):

        residual = x.clone()
        x, attention = self.mha(x, mask=mask)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x.clone()
        x = x.transpose(-1, -2)
        x = self.conv1d(x)
        x = x.transpose(-1, -2)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, attention


class ApricotT(Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_input: int,
        d_static: int,
        max_code: int,
        q: int,
        v: int,
        h: int,
        N: int,
        device: str,
        dropout: float = 0.1,
    ):
        super(ApricotT, self).__init__()

        self.conv1d = Sequential(
            nn.Conv1d(2, d_hidden, 1), nn.ReLU(), nn.Conv1d(d_hidden, d_model, 1)
        )

        self.embedding_variable = Embedding(max_code + 1, d_model)

        self.ffn_static = torch.nn.Sequential(
            torch.nn.Linear(d_static, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        )

        self.encoder_list = ModuleList(
            [
                Encoder(d_model=d_model, d_hidden=d_model * 2, q=q, v=v, h=h)
                for _ in range(N)
            ]
        )

        self.mlp = Sequential(Linear(d_model * 5, d_model), Dropout(dropout))

        self.main1 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.main2 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.main3 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.main4 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense1 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense2 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense3 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense4 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense5 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense6 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense7 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self.dense8 = Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        self._d_input = d_input
        self._d_model = d_model
        self.head = h
        self.transformer_layers = N
        self.device = device

    def forward(self, x, static):
        correct_device = x.device
        values = x[:, :, 2].unsqueeze(-1)
        variables = x[:, :, 1].type(torch.IntTensor).to(correct_device)
        times = x[:, :, 0].unsqueeze(-1)
        static = self.ffn_static(static)

        mask = variables.eq(0)
        inverted_mask = ~mask

        attention_mask = inverted_mask.to(torch.float32)

        values = torch.cat([times, values], dim=2)
        values = values.transpose(-1, -2)
        value = self.conv1d(values)
        value = value.transpose(-1, -2)

        embed = self.embedding_variable.to(correct_device)
        key = embed(variables)
        key = key.to(correct_device)

        encoding = value + key
        encoding_0 = encoding.clone()

        pe = torch.ones_like(encoding).to(correct_device)
        position = times.clone()
        for i in range(encoding.size(0)):
            temp = torch.Tensor(range(0, self._d_model, 2)).to(correct_device)
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position[i].float(), temp)  # shape:[input, d_model/2]
            pe[i, :, 0::2] = torch.sin(temp)
            pe[i, :, 1::2] = torch.cos(temp)

        encoding = encoding + pe

        # Initialize empty lists to store attention scores for each layer
        attention_scores = []

        for encoder in self.encoder_list:
            # Get attention scores from the MHA layer in the encoder
            encoding, attention_score = encoder(encoding, attention_mask)
            attention_scores.append(attention_score)

        encoding = encoding.transpose(-1, -2)

        encoding = torch.topk(encoding, k=5, dim=2)[0]

        encoding = encoding.reshape(
            encoding.size(0), encoding.size(1) * encoding.size(2)
        )

        encoding = self.mlp(encoding)

        encoding = encoding + static

        outputmain1 = self.main1(encoding)
        m = torch.nn.Sigmoid()
        outputmain1 = m(outputmain1)

        outputmain2 = self.main2(encoding)
        m = torch.nn.Sigmoid()
        outputmain2 = m(outputmain2)

        outputmain3 = self.main3(encoding)
        m = torch.nn.Sigmoid()
        outputmain3 = m(outputmain3)

        outputmain4 = self.main4(encoding)
        m = torch.nn.Sigmoid()
        outputmain4 = m(outputmain4)

        output1 = self.dense1(encoding)
        m = torch.nn.Sigmoid()
        output1 = m(output1)

        output2 = self.dense2(encoding)
        m = torch.nn.Sigmoid()
        output2 = m(output2)

        output3 = self.dense3(encoding)
        m = torch.nn.Sigmoid()
        output3 = m(output3)

        output4 = self.dense4(encoding)
        m = torch.nn.Sigmoid()
        output4 = m(output4)

        output5 = self.dense5(encoding)
        m = torch.nn.Sigmoid()
        output5 = m(output5)

        output6 = self.dense6(encoding)
        m = torch.nn.Sigmoid()
        output6 = m(output6)

        output7 = self.dense7(encoding)
        m = torch.nn.Sigmoid()
        output7 = m(output7)

        output8 = self.dense8(encoding)
        m = torch.nn.Sigmoid()
        output8 = m(output8)

        output = torch.cat(
            [output1, output2, output3, output4, output5, output6, output7, output8],
            dim=1,
        )
        outputmain = torch.cat(
            [outputmain1, outputmain2, outputmain3, outputmain4], dim=1
        )

        output = torch.cat([outputmain, output], dim=1)

        return output
