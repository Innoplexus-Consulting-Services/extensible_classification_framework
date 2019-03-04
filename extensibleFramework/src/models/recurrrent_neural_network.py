import torch
from torch import nn
from torch.autograd import Variable
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_dim, batch_size, n_layers=1, rnn_unit="",
                 is_bidiractional=False):

        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_dim)
        self.rnn_unit = rnn_unit
        self.is_bidiractional = is_bidiractional

        if rnn_unit == "GRU":
            self.rnn = nn.GRU(self.embed_dim, self.hidden_size, self.n_layers)
        elif rnn_unit == "LSTM":
            self.rnn = nn.LSTM(self.embed_dim, self.hidden_size, self.n_layers)
        elif rnn_unit == "RNN":
            self.rnn = nn.RNN(self.embed_dim, self.hidden_size, self.n_layers)

    def forward(self, input):
        """

        :param input: MUST BE BATCH FIRST, THIS IS TO MAKE THE SHAPE COMPATIBLE WITH OTHER MODELS
        :param batch_size:
        :return:
        """
        input = input.transpose(0, 1).long()
        embedded = self.embedding(input)
        hidden = self.init_hidden()
        cell = self.init_hidden()
        if (self.rnn_unit == "GRU" or self.rnn_unit == "RNN"):
            output, hidden = self.rnn(embedded, hidden)
        elif (self.rnn_unit == "LSTM"):
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        return hidden

    def init_hidden(self):
        bidiractional_state = (1 if self.is_bidiractional==False else 2)
        result = Variable(torch.Tensor(np.random.rand(self.n_layers * bidiractional_state, self.batch_size, self.hidden_size)))
        # result = torch.nn.init.xavier_normal_(result, gain=1)
        return result
