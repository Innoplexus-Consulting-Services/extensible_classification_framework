
<h1 align="center">Extensible Classification Framework</h1>
---
Extensible Classsification framework is an engineering effort to make a well defined ensemble engine for the text classifcation task. This notebook is an usage guide for the first relese of Extensible framework.

This documentation will cover following points:
- Features of the implementation
- Network Architecture
- Installation and example dataset download
- Usage directions 
    - Processing data
    - Getting custom vectors from fasttext model
    - Defining model config
    - Running experiments
- Modifying  Components
    - For intermediate users
    - For advanced users

# Salient features
1. TorchText based flexible preprocessing
2. Inbuilt custom biomedical tokenizer
3. Inbuilt custom vectorization for biomedical term
4. Accepts pretrained fasttext, Glove and Word2Vec vectors
5. Splinning a run that performes N experiemnts and gives out best model
6. Random best configuration search and report generation
7. Custom saving and loading module
8. Only 3 step to run all experiments
9. End to end customization for intermediate and advanced users

# Network Architecture
![](extensible_framework.png)

*Figure : Showing essential components involved in Extensible Classification framework. It has following components. (All the paramters shown in **bold** can have multiple values, various network will be constructed at run time by using these options.)*

Iterator : 
  
    A Torchtext Iterator, can take variable batch size and vocab size
Embeddings : 
  
- Pretrained fasttext, Glove and Word2Vec vectors
- Accepts Fasttext model for generating custom embeddings at run time

Bi- LSTM with Attention mechnaism: 
  
    Standard Implementation of  LSTM with Attention mechnaism in forward and backward direction. 
    It has folowing configurable paramters
    - final class num
    - rnn_n_layers
    - hidden_size
    - pretrained_weights
    - dropout
    
Convolution Neural Network ([Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) Yoon Kim et. al., Implmemntation) : 
    It has folowing configurable paramters
    - final class num
    - pretrained_weights
    - dropout
    - out_channel_num
    - kernel_sizes
    
Merge Layer
    It combines output from the LSTM and CNN layer and provide it to FFN. The Merge mode can be:
    - Concatenate
    - Substract
    - Add
    - Matmul
    
Feed Forward Layer : 
    It has folowing configurable paramters
    - output class num
    - ffn_activation
    - ffn_dropout
    - perceptron_per_layer
    - final_output_classes
    - Num layers



# Installation
Installation Including following steps
- Clonning repository
- Installation through pip

``` bash
1. clone https://gitlab.innoplexus.de/Innoplexus-Consulting-Services/extensibleFramework.git
2. pip install -U extensibleFramework/dist/extensible_classification_framework-0.0.1-py3-none-any.whl
```

# Usage Directions
`extensible_classification_framework` can be run by folowing 3 steps:
 1. Preprocess
 2. Prepare Vectors
 3. Run training and paramter selection

#### 1. Preprocess 
The input file must be having two column seperated by tab, comma, semicolo or colon. The first colmn should be a class in interger the second column should be text one at each line.
```bash
python preprocess.py --input_file example/data/crude/small_data.tsv --output_destination example/data/processed/ --sep "tab"
```

#### 2. Prepare Vectors

This step is to call a fast text module prepared usiing gensim for vectorization on custom tokens. For this you need to have a pre-trained fasttext model.
```bash
python prepare_vectors.py --model_path example/fasttext_model/fastText.model --train_file example/data/processed/20190401-171735_train.json --test_file example/data/processed/20190401-171735_test.json --vector_output_file example/vectors/vectors.vec
```

#### 3. Run training and paramter selection
Before running the below given step, setup parameters in the `config.py`. A sample config.py looks like as given below:

```python
class parameters:
    def __init__(self):
        """
        Here you may set parameters to be set
        """
        self.cnn_rnn_vocab_size = 0 #len(sentiment_vocab)
        self.cnn_rnn_embed_dim = 100
        self.cnn_rnn_class_num = [200] # configurable
        self.cnn_out_channel_num = [24] # configurable
        self.cnn_kernel_sizes =  [3,4,5] # configurable
        self.rnn_n_layers = [1,2,3] # configurable
        self.rnn_hidden_size =  [128] # configurable
        self.use_pretrained_weights = True
        self.cnn_rnn_weights =  "" # sentiment_vocab.vectors
        self.cnn_rnn_weight_is_trainable =   False
        self.dropout = [0.2] # configurable
        self.batch_size = 32 #batch_size
        self.merge_mode = "CONCAT"
        self.ffn_activation =  "Relu"
        self.ffn_final_output_classes =  2
        self.ffn_perceptron_per_layer = [[100,50, 25]] # configurable
        self.ffn_layer_wise_dropout = 0.2
        self.learning_rate =  [0.2] # configurable
        self.momentum = [0.9] # configurable
        self.device = "" # device
        .
        .
        .
```
---
All the configurable paramter are well indicated, you can provide many configuration for this paramters and all combination will be explored at run time.

```bash
python main.py --train_json example/data/processed/20190401-171735_train.json --test_json example/data/processed/20190401-171735_test.json --embeddigns example/vectors/vectors.vec --epochs 1 --max_token 1000 --device "gpu"
```

# Modifying  Components

## 1. For Intermediate users

Some changes that you might want to do is as given below. **(These changes do not require recompilation and reinstallation)**

1. Changing performance matrix
2. Changing learning rate decay

### 1.1. Changing performance metrics
The existing performance matrix is implemented in `main.py` as : 

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
This function is being called by `test_accuracy_calculator` and `train` funtion for to calculate accuracy. 
You may change this function to calculate F1, Recall and Precision and run your experimetns. 


### 1.2. Changing learning rate decay

Learning rate decay is implemented as: 
```python
if (i != 0 and i%10 == 0 ):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']/2
    print(" === New Learning rate : ", param_group['lr'], " === ")
```
By default learning rate is halved every 10 epochs. You may change this as well


---

## 2. For Advance Users
**These chnages require recompilation and reinstallation**

### 2.1 Changing  something in the core layers 


Any changes made in the package (inside path `extensibleFramework/extensibleFramework/`) requires recompiling and re-installation.
For example I want to change the ontology file which helps in the custom tokenization metrics. The change the existing ontology file following chnages need to be done. 
1. Replace 'resources/ontology_for_tokenizer.list' with new ontology
2. cd to root project folder `/extensibleFramework`
3. Recompile with `python setup.py sdist bdist_wheel`
4. New source will be generated at folder dist/ as `dist/extensible_classification_framework-0.0.1-py3-none-any.whl`
5. Install the new source with pip `pip install -U dist/extensible_classification_framework-0.0.1-py3-none-any.whl`
