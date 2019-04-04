"""
This implementation is based on
Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
For to view learn about entire implementation see notebooks/sentiment_cnn_torchtext.ipynb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class cnn_text(nn.Module):
    def __init__(self,config_object):
        """
        :param vocab_size: total vocabulary size
        :param embed_dim: embedding dimension
        :param class_num: output class number
        :param out_channel_num:
        :param kernel_sizes:  different  size of sliding kernel size; any integer 1-1000
        :param dropout: a float betwenn 0 and 1
        :param stride: will be equla to embed_dim
        """
        super(cnn_text, self).__init__()
        self.config_object = config_object
        self.vocab_size = config_object.cnn_rnn_vocab_size
        self.embed_dim = config_object.cnn_rnn_embed_dim
        self.class_num = config_object.cnn_rnn_class_num
        self.cnn_out_channel_num = config_object.cnn_out_channel_num
        self.kernel_sizes = config_object.cnn_kernel_sizes
        self.dropout = config_object.dropout
        self.stride = config_object.cnn_rnn_embed_dim # stride and embed size are same for this model
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if config_object.use_pretrained_weights:
            self.embed.weight.data.copy_(config_object.cnn_rnn_weights)
            self.embed.weight.requires_grad = config_object.cnn_rnn_weight_is_trainable
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.cnn_out_channel_num, kernel_size=K, stride=self.stride) for K in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.cnn_out_channel_num, self.class_num)

    def forward(self, x):
        # to addd mathematical stability to the model and prevent breakdown due to sequence shorter than 3 words.
        if x.shape[1] <= 5:
            k = x.shape[1]+ (5-x.shape[1])
            x = torch.Tensor(np.pad(x,  ((0,0),(k,k)), 'minimum')).long().to(self.config_object.device)
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = F.relu(self.fc1(x))  # (N, C)
        # logit = torch.softmax(logit, dim=1)
        return logit