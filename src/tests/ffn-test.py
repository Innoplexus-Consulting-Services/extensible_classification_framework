import unittest
import numpy
from src.models.feed_forward_network import *
import torch



def testffn():
    FFN = ffn(activation='Linear', num_layer=3, input_size=100, output_size=2, perceptron_per_layer=[50, 25, 10],
              dropout=[0.2, 0.2, 0.2])
    tensor = torch.Tensor(numpy.random.random([1, 100]))
    return FFN(tensor)

class MyTest(unittest.TestCase):
    def test(self):
        self.assertEqual(testffn().shape, torch.Size([1,2]))