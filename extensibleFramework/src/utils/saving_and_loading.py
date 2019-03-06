import torch
import os
import time

class objectManager():
    def saver(self, model, models_folder,shown_params = False, model_performance_metrics=None):
        """
         model :  A pytorch Model object
         models_folder : A location of mother folder where the main directory will be created.  
         This directory will houses all other directory formed after each experiment,
         Each directory will be uniquely named by timestemp -"%Y%m%d-%H%M%S"
         shown_params :  To see model paramters after each models is saved. Default : False
         model_performance_metrics : Performance matrics can be any python dictiory or string. 
         If the model_performance_metrics is given as {"Accuracy: 88%, "loss" : 0.67}
         then a file will be written as "Model Name : str(timestr) {"Accuracy: 88%, "loss" : 0.67}"
         This will help in easy selction of the model
         
        """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model_snapshot_folder = timestr
        snapshot_directory = os.path.join(models_folder,model_snapshot_folder)
        #making the mother directory where models for the various experiments will be saved
        if not os.path.exists(models_folder):
            os.makedirs(models_folder+"/")
        try:
            # making sub directory where  individual models for each experiment will be saved
            os.makedirs(snapshot_directory)
        except OSError as exception:
            raise
        # making a common file under mother directory where the performance for each model will be recorded
        if model_performance_metrics != None:
            performance_metrics = open("performance_metrics.txt","a+")
            performance_metrics.write("Model Name : "+str(timestr) + "\t"+str(model_performance_metrics)+"\n")
            performance_metrics.flush()
        # saving model and paramters
        torch.save(model.state_dict(),  os.path.join(snapshot_directory, "ensemble_model.ckpt"))
        paramter_file = open( os.path.join(snapshot_directory, "ensemble_model.params.txt"),"w")
        paramter_file.write(str(model.state_dict))
        # showing model parameter after saving
        if shown_params:
            print ("++++++++ Model dumped with following parameters ++++++++")
            print ("--------------------------------------------------------")
            for param_tensor in model.state_dict():
                print(param_tensor, "\t",model.state_dict()[param_tensor].size())  

    def loader(self,model_path):
        """
        Loading existing model
        model_path :  it requires full model path i.e. train/20190306-144431/ensemble_model.ckpt
        """
        
        model = torch.load(model_path)
        return model
