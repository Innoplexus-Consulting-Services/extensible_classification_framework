from extensible_classification_framework.src.models import convolution_neural_network
from extensible_classification_framework.src.models import recurrent_nn_with_attention
from extensible_classification_framework.src.models import feed_forward_network
from extensible_classification_framework.src.models import extra_layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ensemble_model(nn.Module):
    def __init__(self, config_object):
        super(ensemble_model, self).__init__() 
        self.cnn = convolution_neural_network.cnn_text(config_object)
        self.rnn_attention = recurrent_nn_with_attention.RNNAttentionModel(config_object)
        self.merge_layer = extra_layers.MergeAndFlattern(config_object)
        self.ffn = feed_forward_network.ffn(config_object)
    def forward(self, x):
        cnn_output = self.cnn(x)
        rnnAttention_output = self.rnn_attention(x)
        merge_layer_output = self.merge_layer(cnn_output,rnnAttention_output )
        final_output = self.ffn(merge_layer_output)
        return final_output