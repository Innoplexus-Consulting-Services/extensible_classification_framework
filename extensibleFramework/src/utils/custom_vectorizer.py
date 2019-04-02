from gensim.models.fasttext import FastText as FT_gensim
import os
import warnings

class Vectorizer:
    """
    This calss implements the custom vectorizer that is integrated with the torchtext
    """
    def __init__(self,model_path):
        """
        takes the model path where the fasttext model is placed. 
        for this version, the model must be local
        model_path = /data/extensibleFramework/extensibleFramework/embedidngs/fastText.model
        usage : V = Vectorizer(model_path)
        """
        print("Loading Model")
        self.model = FT_gensim.load(model_path)
        self.model
        print("Loaded Model Successfully")
        
    def prepare_vectors(self,vocab, destination_file):
        """
        This function takes all token  present in the vocabulary along with the destination file.
        The destination file will be written with vector for all the tokens
        vocab :  list of all the vocabs
        destination_file : file where the vector needs to be dumped
        """
        if os.path.exists(destination_file):
            warnings.warn("ALREADY EXSTS : "+ destination_file +"   make sure you dont over write previous vectors")
            reply = input('Go ahead and overwrite (y/n)')
            if str(reply).lower() == 'n':
                pass
                return False
            elif  str(reply).lower() == 'y':
                file_pointer = open(destination_file, "w")
                error_count = 0
                unique_tokens = []
                for each_word in vocab:
                    if each_word not in unique_tokens:
                        try:
                            file_pointer.write (str("_".join(each_word.split(" ")))+" "+str(" ".join([str(i) for i in self.model[each_word].tolist()]))+"\n")
                            unique_tokens.append(each_word)
                        except:    
                            error_count = error_count + 1
                return True
            else:
                print ("Your IQ is too low to code this, please specify correct option \n\n Exiting...")
                return False
        else:
            file_pointer = open(destination_file, "w")
            error_count = 0
            unique_tokens = []
            for each_word in vocab:
                if each_word not in unique_tokens:
                    try:
                        file_pointer.write(str(each_word)+" "+str(self.model[each_word])+"\n")
                    except:
                        error_count = error_count + 1
            return True