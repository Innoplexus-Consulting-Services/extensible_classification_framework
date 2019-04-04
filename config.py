class parameters:
    def __init__(self):
        """
        Here you may set parameters to be set
        """
        self.cnn_rnn_vocab_size = 0 #len(sentiment_vocab)
        self.cnn_rnn_embed_dim = 300
        self.cnn_rnn_class_num = [200, 100] # configurable
        self.cnn_out_channel_num = [24, 12, 6] # configurable
        self.cnn_kernel_sizes =  [3,4,5] # configurable
        self.rnn_n_layers = [1,2,3] # configurable
        self.rnn_hidden_size =  [128, 256, 512] # configurable
        self.use_pretrained_weights = True
        self.cnn_rnn_weights =  "" # sentiment_vocab.vectors
        self.cnn_rnn_weight_is_trainable =   False
        self.dropout = [0.2, 0.3, 0.4] # configurable
        self.batch_size = 32 #batch_size
        self.merge_mode = "CONCAT"
        self.ffn_activation =  "Relu"
        self.ffn_final_output_classes =  3
        self.ffn_perceptron_per_layer = [[100,50, 25], [200,100, 50]] # configurable
        self.ffn_layer_wise_dropout = 0.2
        self.learning_rate =  [0.2, 0.1] # configurable
        self.momentum = [0.9, 0.8] # configurable
        self.device = "" # device

    def set_cnn_rnn_weights(self, weights):
        """
        to set weight from main.py 
        """
        self.cnn_rnn_weights = weights

    def set_device(self, hardware):
        """
        to set device from main.py 
        """
        self.device = hardware
        
    def set_cnn_rnn_vocab_size(self, value):
        """
        to set vocab size main.py 
        """
        self.cnn_rnn_vocab_size = value

    def get_batch_size(self):
        """
        to set batch size main.py 
        """
        return self.batch_size