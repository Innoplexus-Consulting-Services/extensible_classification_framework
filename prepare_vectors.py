
import argparse

from torchtext import data, vocab

from extensible_classification_framework.src.utils import (biomedical_tokenizer,
                                           custom_vectorizer, grid_search,
                                           saving_and_loading)


def vectorize(model_path, train_file, test_file, vector_output_file, max_vocab_size = 1000000):
    def tokenize(sentiments):
        return sentiments.lower().split(" ")
    def to_categorical(x):
        x = int(x)
        if x == 1:
            return [0,1]
        if x == 0:
            return [1,0]
    CV = custom_vectorizer.Vectorizer(model_path)
    REVIEW = data.Field(sequential=True , tokenize=tokenize, use_vocab = True, lower=True,batch_first=True)
    LABEL = data.Field(is_target=True,use_vocab = False, sequential=False, preprocessing = to_categorical)
    fields = {'review': ('review', REVIEW), 'label': ('label', LABEL)}
    train_data , test_data = data.TabularDataset.splits(
                                path = '',
                                train = train_file,
                                test = test_file,
                                format = 'json',
                                fields = fields)

    REVIEW.build_vocab(train_data, test_data, max_size=max_vocab_size)
    CV.prepare_vectors(REVIEW.vocab.itos,vector_output_file)
    print("Vocab Size : ",len(REVIEW.vocab))

if __name__=="__main__":
    """
    Example usage :
    python prepare_vectors.py --model_path /data/extensible_classification_framework/extensible_classification_framework/embedidngs/fastText.model --train_file /data/extensible_classification_framework/extensible_classification_framework/data/processed/train.json --test_file /data/extensible_classification_framework/extensible_classification_framework/data/processed/test.json --vector_output_file /data/extensible_classification_framework/extensible_classification_framework/data/processed/vectors.vec
    """
    parser = argparse.ArgumentParser(description='prepare vector using custom vectorizer')
    parser.add_argument('--model_path', help='local path to fasttext model', required = True)
    parser.add_argument('--train_file', help='Train json file path', required = True)
    parser.add_argument('--test_file', help='Test json file path', required = True)
    parser.add_argument('--vector_output_file', help='Output vector file', required = True)
    parser.add_argument('--max_vocab_size', help='maximum vocabulary to be considered', default = 1000000)
    args = parser.parse_args()
    
    vectorize(args.model_path, args.train_file, args.test_file, args.vector_output_file, args.max_vocab_size)

