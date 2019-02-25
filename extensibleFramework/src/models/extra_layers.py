import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class MergeAndFlattern(nn.Module):
    def __init__(self, mode, batch_size):
        """
        This module tales two tensor and converts applies any of the operation like ADD, SUBSTRACT, MULTIPLY and CONCAT

        :param mode: any of ADD, SUBSTRACT, MULTIPLY or  CONCAT
        :param batch_size: any int
        """
        super(MergeAndFlattern, self).__init__()
        self.mode = mode
        self.batch_size = batch_size

    def forward(self, x,y):
        """

        :param input:
        :return:
        """
        if (self.mode ==  "ADD"):
            return x.reshape(self.batch_size, -1) + y.reshape(self.batch_size, -1)
        if (self.mode ==  "SUBSTRACT"):
            return x.reshape(self.batch_size, -1) - y.reshape(self.batch_size, -1)
        if (self.mode == "MULTIPLY"):
            return x.reshape(self.batch_size, -1) * y.reshape(self.batch_size, -1)
        if (self.mode == "CONCAT"):
            return torch.cat((x.reshape(self.batch_size, -1),y.reshape(self.batch_size, -1)),1)