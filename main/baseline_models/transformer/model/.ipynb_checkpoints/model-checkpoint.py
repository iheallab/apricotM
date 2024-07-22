import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Sequential, Dropout, ModuleList


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


class Transformer(Module):
    def __init__(
        self,
        d_model: int,
        q: int,
        v: int,
        h: int,
        N: int,
        device: str,
        dropout: float = 0.1,
    ):
        super(Transformer, self).__init__()

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

        self._d_model = d_model
        self.device = device

    def forward(self, x):

        encoding = x.clone()

        for encoder in self.encoder_list:
            # Get attention scores from the MHA layer in the encoder
            encoding, attention_score = encoder(encoding, None)

        encoding = encoding.transpose(-1, -2)

        encoding = torch.topk(encoding, k=5, dim=2)[0]

        encoding = encoding.reshape(
            encoding.size(0), encoding.size(1) * encoding.size(2)
        )

        output = self.mlp(encoding)

        outputmain1 = self.main1(output)
        m = torch.nn.Sigmoid()
        outputmain1 = m(outputmain1)

        outputmain2 = self.main2(output)
        m = torch.nn.Sigmoid()
        outputmain2 = m(outputmain2)

        outputmain3 = self.main3(output)
        m = torch.nn.Sigmoid()
        outputmain3 = m(outputmain3)

        outputmain4 = self.main4(output)
        m = torch.nn.Sigmoid()
        outputmain4 = m(outputmain4)

        output1 = self.dense1(output)
        m = torch.nn.Sigmoid()
        output1 = m(output1)

        output2 = self.dense2(output)
        m = torch.nn.Sigmoid()
        output2 = m(output2)

        output3 = self.dense3(output)
        m = torch.nn.Sigmoid()
        output3 = m(output3)

        output4 = self.dense4(output)
        m = torch.nn.Sigmoid()
        output4 = m(output4)

        output5 = self.dense5(output)
        m = torch.nn.Sigmoid()
        output5 = m(output5)

        output6 = self.dense6(output)
        m = torch.nn.Sigmoid()
        output6 = m(output6)

        output7 = self.dense7(output)
        m = torch.nn.Sigmoid()
        output7 = m(output7)

        output8 = self.dense8(output)
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
