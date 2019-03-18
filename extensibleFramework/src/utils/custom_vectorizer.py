from gensim.models.fasttext import FastText as FT_gensim
import os
import warnings

class Vectorizer:
    def __init__(self,model_path):
        """
        model_path : /data/extensibleFramework/extensibleFramework/embedidngs/fastText.model
        """
        print("Loading Model")
        self.model = FT_gensim.load(model_path)
        self.model
        print("Loaded Model Successfully")

    def prepare_vectors(self,words, destination_file):
        if os.path.exists(destination_file):
            warnings.warn("ALREADY EXSTS : "+ destination_file +"   make sure you dont over write previous vectors")
            reply = input('Go ahead and overwrite (y/n)')
            if str(reply).lower() == 'n':
                pass
                return False
            elif  str(reply).lower() == 'y':
                file_pointer = open(destination_file, "w")
                error_count = 0
                for each_word in words:
                    try:
                        file_pointer.write (str(each_word)+" "+str(" ".join([str(i) for i in self.model[each_word].tolist()]))+"\n")
                    except:    
                        error_count = error_count + 1
                print("Vectos returned to file : ", destination_file)
                print("Total errors : ",error_count)
                file_pointer.flush()
                file_pointer.close ()
                print("File written successfuly : ", destination_file)
            else:
                print ("Your IQ is too low to code this, please specify correct option \n\n Exiting...")
                return False
        else:
            file_pointer = open(destination_file, "w")
            for each_word in words:
                file_pointer.write(str(each_word)+" "+str(self.model[each_word])+"\n")
            file_pointer.flush()
            file_pointer.close ()
            print("File written successfuly : ", destination_file)
            return True