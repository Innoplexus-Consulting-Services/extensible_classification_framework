from nltk.tokenize import MWETokenizer
import os
dir = os.path.dirname(__file__)
biomedical_entity_filename = os.path.join(dir, 'resources/within_5.txt')

class getToknizer:
    def __init__(self):
        self.tokenizer = MWETokenizer([],separator=' ')
        for i in open(biomedical_entity_filename).read().splitlines():
            self.tokenizer.add_mwe(i.split(" "))
        self.tokenizer

