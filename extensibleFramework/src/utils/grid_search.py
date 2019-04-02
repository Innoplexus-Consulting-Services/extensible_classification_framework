import random
import json

class searchParameters:
    def __init__(self,parameters):
        self.parameters = parameters
        self.chnageable_params = parameters
        
    def random_search(self, max_experiment):
        """
        A function to randomaly take any of the specified configuration and return it for the model builing
        """
        for _ in range(0,max_experiment):
            self.chnageable_params.learning_rate = random.choice(self.parameters.learning_rate),
            self.chnageable_params.momentum = random.choice(self.parameters.momentum)
            self.chnageable_params.cnn_out_channel_num = random.choice(self.parameters.cnn_out_channel_num)
            self.chnageable_params.cnn_rnn_class_num = random.choice(self.parameters.cnn_rnn_class_num)
            self.chnageable_params.rnn_n_layers = random.choice(self.parameters.rnn_n_layers)
            self.chnageable_params.rnn_hidden_size = random.choice(self.parameters.rnn_hidden_size)
            self.chnageable_params.dropout = random.choice(self.parameters.dropout)
            self.chnageable_params.ffn_perceptron_per_layer = random.choice(self.parameters.ffn_perceptron_per_layer)

            # adjustng layer wise droup out according to perceptron per layer
            self.chnageable_params.ffn_layer_wise_dropout = [self.parameters.ffn_layer_wise_dropout for i in range(0,len(self.chnageable_params.ffn_perceptron_per_layer))]

            # adjustng ffn layer numbers according to perceptron per layer
            self.chnageable_params.ffn_num_layer = len(self.chnageable_params.ffn_perceptron_per_layer)
            yield self.chnageable_params
        

    

