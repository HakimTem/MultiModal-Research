import os
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

class MosiDataModule(LightningDataModule): 
    def __init__(self, data_dir): 
        super().__init__()
        
        self.data_dir = data_dir
        self.data_file = None

        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None
        
    def prepare_data(self): 
        self.data_file = os.path.join(self.data_dir, "mosi_raw.pkl")
        if not os.path.exists(self.data_file): 
            raise RuntimeError('MOSI Dataset not found')

    def setup(self, stage=None):
        if stage == "fit" or stage is None: 
            train_data_path = os.path.join(self.data_dir, "mosi_train.dt")
            if not os.path.exists(train_data_path): 
                print(f"  - Creating new train data")
                self.train_data = MOSIDataset(self.data_file, split_type='train')
                torch.save(self.train_data, train_data_path)
            else: 
                print(f"  - Found cached train data")
                self.train_data = torch.load(train_data_path)

            valid_data_path = os.path.join(self.data_dir, "mosi_valid.dt")
            if not os.path.exists(valid_data_path): 
                print(f"  - Creating new valid data")
                self.val_data = MOSIDataset(self.data_file, split_type='valid')
                torch.save(self.val_data, valid_data_path)
            else: 
                print(f"  - Found cached valid data")
                self.val_data = torch.load(valid_data_path)

        if stage == "test" or stage is None: 
            test_data_path = os.path.join(self.data_dir, "mosi_test.dt")
            if not os.path.exists(test_data_path): 
                print(f"  - Creating new test data")
                self.test_data = MOSIDataset(self.data_file, split_type='test')
                torch.save(self.test_data, test_data_path)
            else: 
                print(f"  - Found cached test data")
                self.test_data = torch.load(test_data_path)

    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size=24, shuffle=True, num_workers=4)

    def val_dataloader(self): 
        return DataLoader(self.val_data, batch_size=24, shuffle=False, num_workers=4)
    
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size=24, shuffle=False, num_workers=4)

# MOSI Dataset
class MOSIDataset:
    def __init__(self, dataset_path, split_type='train'):
        super(MOSIDataset, self).__init__()
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

        self.n_modalities = 3  # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (self.meta[index][0], self.meta[index][1],
                self.meta[index][2])
        return X, Y, META