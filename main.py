import argparse
import datetime
import json
import os
import random
import re
import sys
import tarfile
import urllib
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import MWETokenizer
from torchtext import data, vocab
from tqdm import tqdm
from torch import nn

import chakin

import config
from extensible_classification_framework.src.models import (convolution_neural_network,
                                            ensemble, extra_layers,
                                            feed_forward_network, load_config,
                                            recurrent_nn_with_attention)
from extensible_classification_framework.src.utils import (biomedical_tokenizer,
                                           custom_vectorizer, grid_search,
                                           saving_and_loading)


def tokenize(sentiments):
    return TOKENIZER.tokenizer.tokenize(sentiments.split())

def to_categorical(x):
    x = int(x)
    if x == 2:
        return [1,0,0]
    if x == 1:
        return [0,1,0]
    if x == 0:
        return [0,0,1]

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.argmax(preds, dim=1)

    correct = (rounded_preds == torch.argmax(y, dim=1)).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    print(classification_report(torch.argmax(y, dim=1), rounded_preds))
    return acc

def test_accuracy_calculator(model, test_iterator,batch_size,device):
    """
    to calculate the test accuracy
    """
    epoch_acc = 0
    for batch in test_iterator:
        if batch.review.shape[0] ==  batch_size:
            feature, target = batch.review, batch.label
            predictions = model(feature.to(device))            
            acc = binary_accuracy(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
            epoch_acc += acc.item()
    return  epoch_acc / len(test_iterator)

def train(model, iterator, optimizer, criterion, device, batch_size):
    """
    to train the model
    """
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        feature, target = batch.review, batch.label
        if feature.shape[0] == batch_size:   
            optimizer.zero_grad()
            predictions = model(feature.to(device))            
            loss = criterion(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
            loss.backward()
            optimizer.step()
            acc = binary_accuracy(predictions.type(torch.FloatTensor), target.type(torch.FloatTensor))
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return model, epoch_loss / len(iterator), epoch_acc / len(iterator)

def run (train_json, test_json, embeddigns, experiment_output_folder, epochs, num_random_search, max_size, device, GS, TOKENIZER, PARAMETERS, SAL):
    """
    Training various models on the same data
    """

    if device == "gpu":
        device = torch.device("cuda")
    else :
        device = torch.device("cpu")

    # initlializing for paramters fetching
    parma_set= next(GS.random_search(1))
    batch_size = parma_set.batch_size

    # review and label preprocessing
    REVIEW = data.Field(sequential=True , tokenize=tokenize, use_vocab = True, lower=True,batch_first=True)
    LABEL = data.Field(is_target=True,use_vocab = False, sequential=False, preprocessing = to_categorical)
    fields = {'review': ('review', REVIEW), 'label': ('label', LABEL)}
    train_data , test_data = data.TabularDataset.splits(
                                path = '',
                                train = train_json,
                                test = test_json,
                                format = 'json',
                                skip_header=True,
                                fields = fields) 
    vec = vocab.Vectors(name = embeddigns,cache ="./")
    REVIEW.build_vocab(train_data, test_data, max_size=int(max_size), vectors=vec)
    sentiment_vocab = REVIEW.vocab
    print("Length of the Vocab is : ",len(sentiment_vocab))

    # Designing Iterator
    train_iter, test_iter = data.Iterator.splits(
        (train_data, test_data), sort_key=lambda x: len(x.review),
        batch_sizes=(batch_size,batch_size), device=device)

    # setting device, vocab size and vocan vectors to the parameter class
    PARAMETERS = config.parameters()
    PARAMETERS.set_cnn_rnn_vocab_size(len(sentiment_vocab))
    PARAMETERS.set_cnn_rnn_weights(sentiment_vocab.vectors)
    PARAMETERS.set_device(device)
    GS = grid_search.searchParameters(PARAMETERS)

    for parma_set in GS.random_search(int(num_random_search)):
        configuration = load_config.Config(parma_set)
        # initializing ensemble model
        ENSEMBLE_MODEL   = ensemble.ensemble_model(configuration)
        ENSEMBLE_MODEL   = ENSEMBLE_MODEL.to(device)
        # defining optimizer and loss function
        optimizer = torch.optim.SGD(ENSEMBLE_MODEL.parameters(), lr=0.1,momentum=0.9)
        criterion = nn.MSELoss()
        criterion = criterion.to(device)
        
        for i in tqdm(range(int(epochs))):
            if (i != 0 and i%10 == 0 ):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr']/2
                print(" === New Learning rate : ", param_group['lr'], " === ")

            model, epoch_loss, epoch_acc = train(ENSEMBLE_MODEL , train_iter, optimizer, criterion, device, batch_size)

            test_acc = test_accuracy_calculator(model, test_iter, batch_size, device)

            print("Epoch Accuracy : ", epoch_acc," Test Accuracy : ", test_acc, " Epoc Loss : ",epoch_loss)

        del configuration.device # because it is not serilizable
        detailed_params  = json.dumps(configuration, default=lambda x: x.__dict__)
        SAL.saver(ENSEMBLE_MODEL , experiment_output_folder,model_performance_metrics={"Train_accuracy":epoch_acc,"Test accuracy": test_acc, "Epoch Loss":epoch_loss}, detailed_params=detailed_params )
    

if __name__=="__main__":
        # python main.py --train_json /data/extensible_classification_framework/extensible_classification_framework/data/processed/train.json --test_json /data/extensible_classification_framework/extensible_classification_framework/data/processed/test.json --embeddigns /data/extensible_classification_framework/extensible_classification_framework/embedidngs/glove.6B.100d.txt --epochs 1 --max_token 1000 --device = "gpu"

        # creating objects
        SAL = saving_and_loading.objectManager()
        TOKENIZER = biomedical_tokenizer.getToknizer()
        PARAMETERS = config.parameters()
        GS = grid_search.searchParameters(PARAMETERS)

        # fixing seeds and device
        torch.manual_seed(0)
        np.random.seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if you have multiple GPU and if you want to use any perticular GPU then  uncomment below line
        # device = torch.cuda.set_device(1) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        parser = argparse.ArgumentParser(description='Preprocess data for text classification')
        parser.add_argument('--train_json', help='Train File prepared by using preprocess.py', required = True)
        parser.add_argument('--test_json',
                            help='Test File prepared using preprocess.py', required = True)
        parser.add_argument('--embeddigns', help='Embedding file using prepare_vectors.py or pretrained downloaded vectors.', required = True)
        parser.add_argument('--experiment_output_folder', help='Output folder where all the experiments will be saved.', required = True)
        parser.add_argument('--epochs', help='Max epoch for a single model trainig.', required = True, type=int)
        parser.add_argument('--num_random_search', help='Number of random search to be carried out.', required = True, type=int)
        parser.add_argument('--max_token', help='Max token to be considered for vocab generation.', required = False, default = 100000, type=int)
        parser.add_argument('--device', help='CPU or GPU on which computation to be run', required=False, default = "gpu")
        


        args = parser.parse_args()
        run(args.train_json, args.test_json, args.embeddigns, args.experiment_output_folder, args.epochs, args.num_random_search, args.max_token, args.device, GS, TOKENIZER, PARAMETERS, SAL)