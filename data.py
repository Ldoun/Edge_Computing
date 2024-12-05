import torch
from torch.utils.data import  Dataset


def load_data(file):
    pass #data loading func

class DataSet(Dataset):
    def __init__(self, file_list, label=None):
        self.file_list = file_list
        self.label = label

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.label is None:
            return {'data' : torch.tensor(load_data(self.file_list[index]), dtype=torch.float)}
        else:
            return {'data' : torch.tensor(load_data(self.file_list[index]), dtype=torch.float), 
                    'label' : torch.tensor(load_data(self.label[index]), dtype=torch.float)}
        
class Preload_DataSet(Dataset):
    def __init__(self, file_list, label=None):
        self.file_list = file_list
        self.label = label

        self.data = torch.stack([load_data[file] for file in file_list])

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        if self.label is None:
            return {'data' : self.data[index]}
        else:
            return {'data' : self.data[index], 
                    'label' : torch.tensor(load_data(self.label[index]), dtype=torch.float)}