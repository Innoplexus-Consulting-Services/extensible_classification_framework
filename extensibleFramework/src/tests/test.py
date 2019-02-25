

import unittest
import numpy
from extensibleFramework.src.models.feed_forward_network import *
from extensibleFramework.src.models.convolution_neural_network import *
from extensibleFramework.   src.models.recurrrent_neural_network import *

import torch

"""
test cases for FFN module
"""
def testffn():
    FFN = ffn(activation='Linear', num_layer=3, input_size=100, output_size=2, perceptron_per_layer=[50, 25, 10],
              dropout=[0.2, 0.2, 0.2])
    tensor = torch.Tensor(numpy.random.random([1, 100]))
    return FFN(tensor)

# ============================================================================ #
"""
test cases for CNN module
"""


def testcnn():
    CNN = cnn_text(vocab_size = 5, embed_dim  = 10 , class_num =  2, out_channel_num = 8, kernel_sizes = [2,3,4], dropout = 0.5)
    tensor = torch.Tensor(numpy.random.randint(5, size=[8, 32])).long()
    return CNN(tensor)

# =============================================================================== #

input_size = 10
hidden_size = 128
embed_size = 100
batch_size = 8
n_layers = 1


def testrnn():
    ENCODER = RNN(input_size, hidden_size, embed_size, batch_size, n_layers=1, rnn_unit="GRU")
    input = torch.Tensor(numpy.random.random([batch_size, input_size]))
    encoder_output, encoder_hidden = ENCODER(input)
    # print("ENCODER OUTPUT SHAPE : ", encoder_output.shape, "ENCODER HIDDEN STATE SHAPE : ", encoder_hidden.shape)
    return encoder_output, encoder_hidden

# =============================================================================== #

class master_test(unittest.TestCase):
    def test(self):
        self.assertEqual(testrnn()[0].shape, torch.Size([10,8,128]))
        self.assertEqual(testcnn().shape, torch.Size([8,2]))
        self.assertEqual(testffn().shape, torch.Size([1,2]))


