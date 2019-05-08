

import unittest
import numpy
from extensible_classification_framework.src.models.feed_forward_network import *
from extensible_classification_framework.src.models.convolution_neural_network import *
from extensible_classification_framework.   src.models.recurrent_nn_with_attention import *

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


def testrnnAttention():
    AM = RNNAttentionModel(output_size=20, hidden_size=100, vocab_size=64, embedding_length=120,
                           batch_size=8,weights="")
    tensor = torch.Tensor(np.random.randint(5, size=[8, 32])).long()
    return AM(tensor)

# =============================================================================== #

class master_test(unittest.TestCase):
    def test(self):
        self.assertEqual(testrnnAttention().shape, torch.Size([8,20]))
        self.assertEqual(testcnn().shape, torch.Size([8,2]))
        self.assertEqual(testffn().shape, torch.Size([1,2]))


