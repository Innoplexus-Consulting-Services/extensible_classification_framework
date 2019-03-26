class Config:
    def __init__(self,config_object):
        """
        Arguments
        ------------
        cnn_rnn_vocab_size  # Default : Need to set;  The total number of words in the trin data. This is required for to set embeddings layer input size in both cnn and rnn
        cnn_rnn_embed_dim # Default : 100; Embedding dimensions are required for cnn and rnn to set the outpuyt of the embeddings layer 
        cnn_rnn_class_num # Default : 50; output class number from cnn and ffn, this number is generally kept as 50 or 100 then this will be MErged and passed to the FFN
        out_channel_num # Default : 8; Conv2d Output channels for convolution module
        cnn_kernel_sizes # Default [2,3,4] Conv2D kernal/filter size
        rnn_n_layers # number of LSTM
        rnn_hidden_size # hidden untit size for LSTM units
        cnn_rnn_weights # pretrained embeddig weight, must be given as torchtext vocab object
        cnn_rnn_weight_is_trainable # to keep pretrained weight trinable or not
        dropout # A dropout probability for all subnets 
        batch_size # data batch size
        merge_mode # merge mode can be any thing out of any of ADD, SUBSTRACT, MULTIPLY or  CONCAT
        set_ffn_input_size() # ffn layer input size
        device =  CPU or GPU
        """

        self.cnn_rnn_vocab_size = config_object.cnn_rnn_vocab_size  # Default : Need to set;  The total number of words in the trin data. This is required for to set embeddings layer input size in both cnn and rnn
        self.cnn_rnn_embed_dim = config_object.cnn_rnn_embed_dim # Default : 100; Embedding dimensions are required for cnn and rnn to set the outpuyt of the embeddings layer 
        self.cnn_rnn_class_num = config_object.cnn_rnn_class_num # Default : 50; output class number from cnn and ffn, this number is generally kept as 50 or 100 then this will be MErged and passed to the FFN
        self.cnn_out_channel_num = config_object.cnn_out_channel_num # Default : 8; Conv2d Output channels for convolution module
        self.cnn_kernel_sizes = config_object.cnn_kernel_sizes # Default [2,3,4] Conv2D kernal/filter size
        self.rnn_n_layers= config_object.rnn_n_layers # number of LSTM
        self.rnn_hidden_size = config_object.rnn_hidden_size # hidden untit size for LSTM units
        self.use_pretrained_weights  = config_object.use_pretrained_weights
        self.cnn_rnn_weights = config_object. cnn_rnn_weights # pretrained embeddig weight, must be given as torchtext vocab object
        self.cnn_rnn_weight_is_trainable = config_object.cnn_rnn_weight_is_trainable # to keep pretrained weight trinable or not
        self.dropout = config_object.dropout # A dropout probability for all subnets 
        self.batch_size  = config_object.batch_size # data batch size
        self.merge_mode = config_object.merge_mode # merge mode can be any thing out of any of ADD, SUBSTRACT, MULTIPLY or  CONCAT
        self.ffn_input_size  = self.set_ffn_input_size() # ffn layer input size
        self.ffn_activation = config_object.ffn_activation # Any of the above Relu, Selu, Tanh, Sigmoid
        self.ffn_final_output_classes = config_object.ffn_final_output_classes # Default : 2 for binary classificatio; this id the final output from taking input from cnn ans rnn
        self.ffn_num_layer = config_object.ffn_num_layer; # Default can be 2-3 or more, Number of layers in the ffn module
        self.ffn_perceptron_per_layer = config_object.ffn_perceptron_per_layer # perceptron in each layers list [50, 25]
        self.ffn_layer_wise_dropout = config_object.ffn_layer_wise_dropout # layer wise droupout for eachlayer in ffn a list with probabilities [0.2, 0.2]
        self.device = config_object.device # csn be defined as device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # checking sanity of provided parameters
        self.sanity_check()

        # learning rate decay related parameters
        # self.learning_rate
        # self.current_epoch
        # self.decrease_by
        # self.after_every_epoch
        

    def set_ffn_input_size(self):
        """
        setting ffn input size according to the merge_mode and cnn_rnn_class_num
        """
        if (self.merge_mode ==  "ADD"):
            return self.cnn_rnn_class_num
        if (self.merge_mode ==  "SUBSTRACT"):
            return self.cnn_rnn_class_num
        if (self.merge_mode == "MULTIPLY"):
            return self.cnn_rnn_class_num
        if (self.merge_mode == "CONCAT"):
            return self.cnn_rnn_class_num*2
    def sanity_check(self):
        """
        to check if all variables are initilized properly without conflict
        """
        assert (len(self.ffn_perceptron_per_layer) == len(self.ffn_layer_wise_dropout)), "Perceptron perlayer and number of droupout mismath"
        assert (self.ffn_num_layer == len(self.ffn_perceptron_per_layer)), "Number of layer and length of perceptron perlayer do not match"
        assert (self.ffn_num_layer == len(self.ffn_layer_wise_dropout)), "Number of layer and number of droupout mismath"
