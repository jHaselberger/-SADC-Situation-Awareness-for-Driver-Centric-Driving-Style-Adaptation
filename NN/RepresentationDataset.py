
from torch.utils.data import Dataset
    
class RepresentationDataset(Dataset):
    def __init__(self, representations, labels=None): 
        self.X = representations
        self.y = labels
        self.has_labels = False if labels is None else True

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"X": self.X[idx], "y": self.y[idx]} if self.has_labels else {"X": self.X[idx]}