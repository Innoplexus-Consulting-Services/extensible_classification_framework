import torch
from collections import OrderedDict
import numpy
from torch.autograd import Variable
from torch import nn
import random




class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, batch_size, n_layers=1, rnn_unit = ""):

        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn_unit = rnn_unit

        if rnn_unit ==  "GRU":
            self.rnn = nn.GRU(self.embed_size, self.hidden_size,self.n_layers )
        elif rnn_unit ==  "LSTM":
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size,self.n_layers )
        elif rnn_unit ==  "RNN":
            self.rnn = nn.RNN(self.embed_size, self.hidden_size,self.n_layers )

    def forward(self, input):
        """

        :param input: MUST BE BATCH FIRST
        :param batch_size:
        :return:
        """
        input = input.transpose(0,1).long() # making batch last
        embedded = self.embedding(input) #Input =  64 --->  #Output  [64,1 ]
        hidden = self.init_hidden()
        cell = self.init_hidden()

        if (self.rnn_unit == "GRU" or self.rnn_unit == "RNN"):
            output, hidden = self.rnn(embedded, hidden)
        elif (self.rnn_unit == "LSTM"):
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        return output, hidden #Output 1, 64, 128  #encoder Hidden = 1, 64, 128

    def init_hidden(self):
        result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        result = torch.nn.init.xavier_normal_(result,gain=1)
        return result #Output 1, 64, 128


input_size = 10
hidden_size = 128
embed_size = 100
batch_size = 8
n_layers = 1
