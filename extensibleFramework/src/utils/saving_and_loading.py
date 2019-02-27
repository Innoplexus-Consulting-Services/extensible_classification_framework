import torch

class objectManager():
    def saver(self, model, PATH):
        torch.save(model.state_dict(), PATH)
        print ("++++++++ Model dumped with following parameters ++++++++")
        print ("--------------------------------------------------------")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t",model.state_dict()[param_tensor].size())        
    def loader(self,PATH):
        model = torch.load(PATH)
        return model
