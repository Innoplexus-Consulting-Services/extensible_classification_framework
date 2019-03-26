class parameters:
    def __init__(self):
        self.cnn_rnn_vocab_size = 0 #len(sentiment_vocab)
        self.cnn_rnn_embed_dim = 100
        self.cnn_rnn_class_num = [200]
        self.cnn_out_channel_num = [24]
        self.cnn_kernel_sizes =  [3,4,5]
        self.rnn_n_layers = [1,2,3]
        self.rnn_hidden_size =  [128]
        self.use_pretrained_weights = True
        self.cnn_rnn_weights =  "" # sentiment_vocab.vectors
        self.cnn_rnn_weight_is_trainable =   False
        self.dropout = [0.2]
        self.batch_size = 32 #batch_size
        self.merge_mode = "CONCAT"
        self.ffn_activation =  "Relu"
        self.ffn_final_output_classes =  2
        self.ffn_perceptron_per_layer = [[100,50, 25]]
        self.ffn_layer_wise_dropout = 0.2
        self.learning_rate =  [0.2]
        self.momentum = [0.9]
        self.device = "" # device

    def set_cnn_rnn_weights(self, weights):
        self.cnn_rnn_weights = weights

    def set_device(self, device):
        self.device = device
