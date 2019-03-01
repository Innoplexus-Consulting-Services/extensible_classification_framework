
# Extensible Framwork
Extensible framework is an engineering effort to make a well defined ensemble engine for the text classifcation task.

This notebook is an usage guide for the first relese of Extensible framework.

This documentation will cover following points:
1. Installation and example dataset download
2. Preprocessing with Torchtext
3. Defining model config
4. Defining data iterator
5. Definning emsemble model
6. Defining Model  Training/Test Functions
7. Defining optimizer, loss and learning rate
8. Running the training for desired epochs
9. Model Save ans load mechanism

## Installation and example data download

Installtion support  in not planned in this release but you can always use it by adding the module to system Path


```python
import sys
sys.path.append('/data/extensibleFramework/')
```


```python
from extensibleFramework.src.models import convolution_neural_network
from extensibleFramework.src.models import recurrent_nn_with_attention
from extensibleFramework.src.models import feed_forward_network
from extensibleFramework.src.models import extra_layers
from extensibleFramework.src.utils import saving_and_loading
from extensibleFramework.src.models import config
```


```python
import pandas as pd
import chakin
import matplotlib.pyplot as plt
from torchtext import data
import nltk
import json
from torchtext import vocab
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import re
import pandas as pd
import os
import numpy as np
import sys
import random
import tarfile
import urllib
from torchtext import data
import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
data_frame = pd.read_csv("labeledTrainData.tsv",sep="\t")
data_frame = data_frame[['review', 'sentiment']]
data_block  = data_frame.values
```

Splitting data in to train and test


```python
split = 0.80 # train test split
random.shuffle(data_block)
train_file = open('train.json', 'w')
test_file = open('test.json', 'w')
for i in  range(0,int(len(data_block)*split)):
    train_file.write(str(json.dumps({'review' : data_block[i][0], 'label' : data_block[i][1]}))+"\n")
for i in  range(int(len(data_block)*split),len(data_block)):
    test_file.write(str(json.dumps({'review' : data_block[i][0], 'label' : data_block[i][1]}))+"\n")
```

## Preprocessing with Torchtext
Please trfer to Torchtext module for detailed usage : [http://torchtext.readthedocs.io/](http://torchtext.readthedocs.io/)


```python
def tokenize(sentiments):
    return nltk.tokenize.word_tokenize(sentiments)
def to_categorical(x):
    x = int(x)
    if x == 1:
        return [0,1]
    if x == 0:
        return [1,0]
```

### Defining fields


```python
SENTIMENT = data.Field(sequential=True , tokenize=tokenize, use_vocab = True, lower=True,batch_first=True)
LABEL = data.Field(is_target=True,use_vocab = False, sequential=False, preprocessing = to_categorical)
fields = {'sentiment': ('sentiment', SENTIMENT), 'label': ('label', LABEL)}
train_data , test_data = data.TabularDataset.splits(
                            path = '',
                            train = 'train.json',
                            test = 'test.json',
                            format = 'json',
                            fields = fields)

```


```python
# chakin.search(lang='English')
```


```python
# chakin.download(number=12, save_dir = "./")
```


```python
# !unzip glove.6B.zip
```

### Building Vocab on the basis of fasttext vectors


```python
vec = vocab.Vectors(name = "wiki.en.vec",cache = "./")
```


```python
SENTIMENT.build_vocab(train_data, test_data, vectors=vec)
```


```python
sentiment_vocab = SENTIMENT.vocab
```

## Defining Model Configuration


```python
# defining config object 
cnn_rnn_vocab_size = len(SENTIMENT.vocab)
cnn_rnn_embed_dim = 300
cnn_rnn_class_num = 50
out_channel_num = 8
cnn_kernel_sizes = [3,4,5]
rnn_n_layers = 1
rnn_hidden_size = 128
use_pretrained_weights = True
cnn_rnn_weights = sentiment_vocab.vectors
cnn_rnn_weight_is_trainable =  True
dropout = 0.2
batch_size = 32
merge_mode = "CONCAT"
ffn_activation = "Relu"
ffn_final_output_classes = 2
ffn_num_layer = 2 
ffn_perceptron_per_layer = [50, 25]
ffn_layer_wise_dropout = [0.2,0.2]
device = device
# constructing model config object
config = config.Config(cnn_rnn_vocab_size, cnn_rnn_embed_dim, cnn_rnn_class_num, out_channel_num, \
        cnn_kernel_sizes, rnn_n_layers, rnn_hidden_size, use_pretrained_weights,cnn_rnn_weights, cnn_rnn_weight_is_trainable,\
            dropout, batch_size, merge_mode, ffn_activation, ffn_final_output_classes, ffn_num_layer, ffn_perceptron_per_layer,\
                       ffn_layer_wise_dropout, device)

```

## Defining Data Iterator


```python
train_iter, test_iter = data.Iterator.splits(
        (train_data, test_data), sort_key=lambda x: len(x.sentiment),
        batch_sizes=(batch_size,batch_size), device=device)
```

### Checking Iterator


```python
for batch in test_iter:
    feature, target = batch.sentiment, batch.label
    print(feature.data.shape, target.data.shape)
    break
```

### Checking total vocab size. With this vocabs we will initialize the embedding layer


```python
sentiment_vocab.vectors.shape
```

## Defining Ensemble Model


```python
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
```


```python
EM  = ensemble_model(config)
EM  = EM.to(device)
```

Checking model with example dataset


```python
cnn_input = torch.Tensor(np.random.randint(5, size=[8, 10])).long().to(device)
rnn_input = torch.Tensor(np.random.random([8, 10])).long().to(device) #batch_size, input_size
```

## Defining Model  Training/Test Functions


```python
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == torch.argmax(y, dim=1)).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc
```


```python
def train(model, iterator, optimizer, criterion):
    """
    To iterate over given dataset for one epoch
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        feature, target = batch.sentiment, batch.label
        optimizer.zero_grad()
        predictions = model(feature.to(device))            
        loss = criterion(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        acc = binary_accuracy(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return model, epoch_loss / len(iterator), epoch_acc / len(iterator)
```


```python
def test_accuracy_calculator(model, test_iterator):
    """
    Function to canculate test data accuracy
    
    """
    epoch_acc = 0
    for batch in test_iterator:
        if batch.sentiment.shape[0] ==  32:
            feature, target = batch.sentiment, batch.label
            predictions = model(feature.to(device))            
            acc = binary_accuracy(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
            epoch_acc += acc.item()
    return  epoch_acc / len(test_iterator)
```

# Defining optimizer, loss and learning rate


```python
optimizer = torch.optim.SGD(EM.parameters(), lr=0.1,momentum=0.9)
criterion = nn.MSELoss()
criterion = criterion.to(device)
```

# Running the training for desired epochs


```python
epochs  = 100
log_interval = 1
loss = []
accuracy = []
test_accuracy = []
for i in tqdm(range(epochs)):
    if (i != 0 and i%10 == 0 ):
        # halving learning rate after every 10 epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']/2
        print (" === New Learning rate : ", param_group['lr'], " === ")
    
    model, epoch_loss, epoch_acc = train(EM, train_iter, optimizer, criterion)
    
    test_acc = test_accuracy_calculator(model, test_iter)
    accuracy.append(epoch_acc)
    loss.append(epoch_loss)
    test_accuracy.append(test_acc)
    print(epoch_acc,test_acc,epoch_loss)
```

# Model Save ans load

## Initlizing model save and load object


```python
SAL = saving_and_loading.objectManager()
```

Save the model 


```python
SAL.saver(EM, "./EM.ckpt")
```

Load the model 


```python
EM = SAL.loader("./EM.ckpt")
```
