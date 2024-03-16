import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Sequential, Dropout, ModuleList


class GRU(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        device: str,
    ):
        super(GRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.main1 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.main2 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.main3 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.main4 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense1 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense2 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense3 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense4 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense5 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense6 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense7 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self.dense8 = Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

        self._d_model = hidden_size
        self.device = device

    def forward(self, x):

        encoding, output = self.gru(x)

        output = output[-1]

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
