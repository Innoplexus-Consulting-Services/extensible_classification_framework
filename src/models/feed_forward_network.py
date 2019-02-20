import torch
from collections import OrderedDict
import numpy


class ffn(torch.nn.Module):
    def __init__(self, activation:str, num_layer : int, input_size : int, output_size : int, perceptron_per_layer : list , dropout = list):
        """
        Feed Forward Network class
        :param activation: String : Any of the above Relu, Selu, Tanh, Sigmoid
        :param num_layer: Integer List
        :param input_size: Integer
        :param output_size: Integer
        :param perceptron_per_layer: Integer List
        :param dropout: Float list , values between 0-1

        Usage : FFN = ffn(activation='Relu', num_layer=3, input_size=100, output_size=2, perceptron_per_layer=[50, 25, 10], dropout=[0.2, 0.2, 0.2])
                tensor = torch.Tensor(numpy.random.random([1, 100]))
                print (FFN(tensor))
        """
        super(ffn, self).__init__()
        self.activation =activation
        self.num_layer =num_layer
        self.input_size =  input_size
        self.output_size =  output_size
        self.perceptron_per_layer =perceptron_per_layer
        self.dropout = dropout

        assert num_layer == len(perceptron_per_layer)
        assert num_layer == len(dropout)

        if (activation == 'Relu'):
            self.activation = torch.nn.ReLU()
        elif (activation == 'Selu'):
            self.activation = torch.nn.SELU()
        elif (activation == 'Tanh'):
            self.activation = torch.nn.Tanh()
        elif (activation == 'Sigmoid'):
            self.activation = torch.nn.Sigmoid()

        self.linear_stacking()

    def linear_stacking(self):
        """
        linear stack layers with droupout and actiivations
        :return: torch.nn.Sequential object
        """
        self.layers = OrderedDict()
        self.layers['input'] = torch.nn.Linear(self.input_size, self.perceptron_per_layer[0])
        if self.activation != 'Linear':
            self.layers['activation0'] = self.activation
        self.layers['Dropout' + str(0)] = torch.nn.Dropout(self.dropout[0])
        for i in range(0, len(self.perceptron_per_layer) - 1):
            self.layers['Linear' + str(i)] = torch.nn.Linear(self.perceptron_per_layer[i], self.perceptron_per_layer[i + 1])
            if self.activation != 'Linear':
                self.layers['activation'+str(i+1)] = self.activation
            if self.dropout != []:
                self.layers['Dropout'+str(i+1)]  =  torch.nn.Dropout(self.dropout[i+1])
        self.layers['output'] = torch.nn.Linear(self.perceptron_per_layer[-1], self.output_size)
        self.network = torch.nn.Sequential(self.layers)
        return self.network

    def forward(self, X ):
        """
        Combine and excecute the network
        :param X:
        :return:
        """
        return self.network(X)
