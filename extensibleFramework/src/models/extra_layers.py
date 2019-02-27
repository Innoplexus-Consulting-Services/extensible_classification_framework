import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class MergeAndFlattern(nn.Module):
    def __init__(self,config_object):
        """
        This module tales two tensor and converts applies any of the operation like ADD, SUBSTRACT, MULTIPLY and CONCAT

        :param mode: any of ADD, SUBSTRACT, MULTIPLY or  CONCAT
        :param batch_size: any int
        """
        super(MergeAndFlattern, self).__init__()
        self.merge_mode = config_object.merge_mode
        self.batch_size = config_object.batch_size

    def forward(self, x,y):
        """

        :param input:
        :return:
        """
        if (self.merge_mode ==  "ADD"):
            return x.reshape(self.batch_size, -1) + y.reshape(self.batch_size, -1)
        if (self.merge_mode ==  "SUBSTRACT"):
            return x.reshape(self.batch_size, -1) - y.reshape(self.batch_size, -1)
        if (self.merge_mode == "MULTIPLY"):
            return x.reshape(self.batch_size, -1) * y.reshape(self.batch_size, -1)
        if (self.merge_mode == "CONCAT"):
            return torch.cat((x.reshape(self.batch_size, -1),y.reshape(self.batch_size, -1)),1)

def adjust_learning_rate(optimizer, learning_rate, current_epoch, decrease_by, after_every_epoch):
    if (current_epoch != 0 and current_epoch%after_every_epoch == 0 ):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/decrease_by
            print (" === New Learning rate : ", param_group['lr'], " === ")