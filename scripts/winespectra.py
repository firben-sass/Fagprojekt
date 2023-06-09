import torch.utils.data as data 

class WineSpectra(data.Dataset):
    def __init__(self, array):
        self.array = array
        super().__init__()
    def __len__(self):
        return len(self.array)
    def __getitem__(self,index):
        return self.array[index]