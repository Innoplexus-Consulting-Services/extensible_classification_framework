import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


class RNNAttentionModel(torch.nn.Module):
    def __init__(self,config_object):
        """
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        AM = RNNAttentionModel(output_size=20, hidden_size=100, vocab_size=64, embedding_length=120,
                                   batch_size=8,weights="")
        tensor = torch.Tensor(np.random.randint(5, size=[8, 32])).long()
        print(AM(tensor).shape)

        """
        super(RNNAttentionModel, self).__init__()

        self.batch_size = config_object.batch_size
        self.output_size = config_object.cnn_rnn_class_num
        self.hidden_size = config_object.rnn_hidden_size
        self.vocab_size = config_object.cnn_rnn_vocab_size
        self.embedding_length = config_object.cnn_rnn_embed_dim
        self.device = config_object.device

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_length)
        if bool(config_object.use_pretrained_weights) == True:
            self.word_embeddings.weight.data.copy_(config_object.cnn_rnn_weights)
            self.word_embeddings.weight.requires_grad = config_object.cnn_rnn_weight_is_trainable


        # self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.output_size)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_sentences):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """

        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)
        if self.batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(self.device)
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(self.device)
        else:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(self.device)
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(self.device)

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (
        h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)

        return logits



