"""
test cases for FFN module
"""

import unittest
import numpy
from src.models.feed_forward_network import *
from src.models.convolution_neural_network import *

import torch


def testffn():
    FFN = ffn(activation='Linear', num_layer=3, input_size=100, output_size=2, perceptron_per_layer=[50, 25, 10],
              dropout=[0.2, 0.2, 0.2])
    tensor = torch.Tensor(numpy.random.random([1, 100]))
    return FFN(tensor)

class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(testffn().shape, torch.Size([1,2]))

# ============================================================================ #
"""
test cases for CNN module
"""


def testcnn():
    CNN = cnn_text(vocab_size = 5, embed_dim  = 10 , class_num =  2, out_channel_num = 8, kernel_sizes = [2,3,4], dropout = 0.5)
    tensor = torch.Tensor(numpy.random.randint(5, size=[8, 32])).long()
    return CNN(tensor)

class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(testcnn().shape, torch.Size([8,2]))


