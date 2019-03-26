import datetime
import json
import os
import random
import re
import sys
import tarfile
import urllib
import argparse

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import MWETokenizer
from torchtext import data, vocab
from tqdm import tqdm

import chakin
from extensibleFramework.src.models import (load_config, convolution_neural_network,
                                            extra_layers, feed_forward_network,
                                            recurrent_nn_with_attention, ensemble)
from extensibleFramework.src.utils import (biomedical_tokenizer,
                                           custom_vectorizer, grid_search,
                                           saving_and_loading)
from extensibleFramework import config
sys.path.append('/data/extensibleFramework/')


# creating objects
SAL = saving_and_loading.objectManager()
TOKENIZER = biomedical_tokenizer.getToknizer()
PARAMETERS = config.parameters()
SP = grid_search.searchParameters(PARAMETERS)


# fixing seeds and device
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# few paramters fetching
parma_set= next(SP.random_search(1))
batch_size = parma_set.batch_size

def tokenize(sentiments):
    return TOKENIZER.tokenizer.tokenize(sentiments.split())

def to_categorical(x):
    x = int(x)
    if x == 1:
        return [0,1]
    if x == 0:
        return [1,0]

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == torch.argmax(y, dim=1)).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def test_accuracy_calculator(model, test_iterator):
    epoch_acc = 0
    for batch in test_iterator:
        if batch.review.shape[0] ==  32:
            feature, target = batch.review, batch.label
            predictions = model(feature.to(device))            
            acc = binary_accuracy(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
            epoch_acc += acc.item()
    return  epoch_acc / len(test_iterator)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        feature, target = batch.review, batch.label
        optimizer.zero_grad()
        predictions = model(feature.to(device))            
        loss = criterion(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        acc = binary_accuracy(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return model, epoch_loss / len(iterator), epoch_acc / len(iterator)

def main ():
    REVIEW = data.Field(sequential=True , tokenize=tokenize, use_vocab = True, lower=True,batch_first=True)
    LABEL = data.Field(is_target=True,use_vocab = False, sequential=False, preprocessing = to_categorical)
    fields = {'review': ('review', REVIEW), 'label': ('label', LABEL)}
    train_data , test_data = data.TabularDataset.splits(
                                path = '',
                                train = '../data/processed/train.json',
                                test = '../data/processed/test.json',
                                format = 'json',
                                fields = fields) 
    vec = vocab.Vectors(name = "../embedidngs/glove.6B.100d.txt",cache = "./")
    REVIEW.build_vocab(train_data, test_data, max_size=400000, vectors=vec)
    sentiment_vocab = REVIEW.vocab
    print("Length of the Vocab is : ",len(sentiment_vocab))

    # Assign vocab and vector to the config object

    ##############################################
    train_iter, test_iter = data.Iterator.splits(
        (train_data, test_data), sort_key=lambda x: len(x.review),
        batch_sizes=(batch_size,batch_size), device=device)

    epochs  = 100
    loss = []
    accuracy = []
    test_accuracy = []
    for parma_set in SP.random_search(1):
        configuration = load_config.Config(parma_set)
        ENSEMBLE_MODEL   = ensemble.ensemble_model(configuration)
        ENSEMBLE_MODEL   = ENSEMBLE_MODEL.to(device)
        optimizer = torch.optim.SGD(ENSEMBLE_MODEL.parameters(), lr=0.1,momentum=0.9)
        criterion = nn.MSELoss()
        criterion = criterion.to(device)
        
        for i in tqdm(range(epochs)):
            if (i != 0 and i%10 == 0 ):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/2
                print(" === New Learning rate : ", param_group['lr'], " === ")

            model, epoch_loss, epoch_acc = train(ENSEMBLE_MODEL , train_iter, optimizer, criterion)

            test_acc = test_accuracy_calculator(model, test_iter)
            accuracy.append(epoch_acc)
            loss.append(epoch_loss)
            test_accuracy.append(test_acc)
            print(epoch_acc,test_acc,epoch_loss)

        del configuration.device # because it is not serilizable
        detailed_params  = json.dumps(configuration, default=lambda x: x.__dict__)
        SAL.saver(ENSEMBLE_MODEL , "train",model_performance_metrics={"Train_accuracy":epoch_acc,"Test accuracy": test_acc, "Epoch Loss":epoch_loss}, detailed_params=detailed_params )
    

if __name__=="__main__":
    seperator = {"tab": "\t", "comma":",", "colon":":", "semicolon":";"}
    parser = argparse.ArgumentParser(description='Preprocess data for text classification')
    parser.add_argument('--input_file', help='Input file having label and review seperated by delimiter.', required = True)
    parser.add_argument('--output_destination',
                        help='Destination folder where preprocessed file will be written.', required = True)
    parser.add_argument('--sep', help='delimiter according to file structure. options: "tab" or "comma" or "colon" or "semicolon"', required = True)
    parser.add_argument('--split_ratio', help='split ratio of the train file, any float value', default = 0.7)
    args = parser.parse_args()
    
    to_json(args.input_file, args.output_destination, seperator[args.sep], args.split_ratio)
