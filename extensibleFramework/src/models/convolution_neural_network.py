"""
This implementation is based on
Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).
For to view learn about entire implementation see notebooks/sentiment_cnn_torchtext.ipynb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class cnn_text(nn.Module):
    def __init__(self, vocab_size :int, embed_dim : int, class_num: int, out_channel_num: int, kernel_sizes:list, dropout:float):
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
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.out_channel_num = out_channel_num
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.stride = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=out_channel_num, kernel_size=K, stride=self.stride) for K in self.kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * out_channel_num, class_num)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = F.relu(self.fc1(x))  # (N, C)
        logit = torch.softmax(logit, dim=1)
        return logit