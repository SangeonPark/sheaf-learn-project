from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional
import os
import pickle
#import torch_sparse
BATCH_SIZE=1

class SheafDataModule(LightningDataModule):
    """docstring for SheafDataModule"""
    def __init__(self, file_dict, batch_size :int = BATCH_SIZE):
        super(SheafDataModule, self).__init__()
        #self.num_types = len(file_dict)
        self.file_dict = file_dict
        #self.data_dir = data_dir
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    #def prepare_data(self):
        #MNIST(os.getcwd(), train=True, download=True)
        #MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.sheaf_train = torch.load(os.path.join(self.file_dict['train']))
            self.sheaf_val   = torch.load(os.path.join(self.file_dict['val']))
            #self.sheaf_train, self.sheaf_val = random_split(sheaf_train, [280000, 40000])
        if stage == "test":
            self.sheaf_test = torch.load(os.path.join(self.file_dict['test']))
        if stage == "predict":
            self.sheaf_predict = torch.load(os.path.join(self.file_dict['predict']))

    # return the dataloader for each split
    def train_dataloader(self):
        sheaf_train = DataLoader(self.sheaf_train, batch_size=self.batch_size,shuffle=True,num_workers=4)
        return sheaf_train

    def val_dataloader(self):
        sheaf_val = DataLoader(self.sheaf_val, batch_size=self.batch_size,num_workers=4)
        return sheaf_val

    def test_dataloader(self):
        sheaf_test = DataLoader(self.sheaf_test, batch_size=self.batch_size)
        return sheaf_test

    def predict_dataloader(self):
        sheaf_predict = DataLoader(self.sheaf_predict, batch_size=self.batch_size)
        return sheaf_predict



class SheafDataModule_ForGraphs(LightningDataModule):
    """docstring for SheafDataModule"""
    def __init__(self, fixed_split, total_size, batch_size :int = BATCH_SIZE):
        super(SheafDataModule_ForGraphs, self).__init__()
        #self.num_types = len(file_dict)
        self.data = fixed_split
        self.batch_size = batch_size
        self.total_size = total_size


    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    #def prepare_data(self):
        #MNIST(os.getcwd(), train=True, download=True)
        #MNIST(os.getcwd(), train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        # transforms
        # split dataset
        #if stage in (None, "fit"):
        #    self.sheaf_train = SheafDataset_ForGraphs(self.data.x[data.train_mask], self.data.y[data.train_mask], self.total_size)
        #    self.sheaf_val = SheafDataset_ForGraphs(self.data.x[data.val_mask], self.data.y[data.val_mask], self.total_size)
        #    #self.sheaf_train, self.sheaf_val = random_split(sheaf_train, [280000, 40000])
        #if stage == "test":
        #    self.sheaf_test = SheafDataset_ForGraphs(self.data.x[data.test_mask], self.data.y[data.test_mask], self.total_size)
        #if stage == "predict":
        #    self.sheaf_predict = SheafDataset_ForGraphs(self.data.x[data.test_mask], self.data.y[data.test_mask], self.total_size)

        if stage in (None, "fit"):
            self.sheaf_train = SheafDataset_ForGraphs(self.data.x, self.data.y, self.total_size)
            self.sheaf_val = SheafDataset_ForGraphs(self.data.x, self.data.y, self.total_size)
            #self.sheaf_train, self.sheaf_val = random_split(sheaf_train, [280000, 40000])
        if stage == "test":
            self.sheaf_test = SheafDataset_ForGraphs(self.data.x, self.data.y ,self.total_size)
        if stage == "predict":
            self.sheaf_predict = SheafDataset_ForGraphs(self.data.x, self.data.y, self.total_size)
    # return the dataloader for each split
    def train_dataloader(self):
        sheaf_train = DataLoader(self.sheaf_train, batch_size=self.batch_size,shuffle=True,num_workers=4)
        return sheaf_train

    def val_dataloader(self):
        sheaf_val = DataLoader(self.sheaf_val, batch_size=self.batch_size,num_workers=4)
        return sheaf_val

    def test_dataloader(self):
        sheaf_test = DataLoader(self.sheaf_test, batch_size=self.batch_size)
        return sheaf_test

    def predict_dataloader(self):
        sheaf_predict = DataLoader(self.sheaf_predict, batch_size=self.batch_size)
        return sheaf_predict


class SheafDataset_ForGraphs(torch.utils.data.Dataset):
    """It the sheaf"""
    def __init__(self, x_data, label, total_size: int):
        super(SheafDataset_ForGraphs, self).__init__()
        self.x_data = x_data
        self.label  = label
        self.total_size = total_size

    def __len__(self):
        #return self.sheaf_data.size(0)
        # HACKY SOLUTION
        return self.total_size

    def __getitem__(self, index):
        return self.x_data, self.label




class SheafDataset(torch.utils.data.Dataset):
    """It the sheaf"""
    def __init__(self, from_file, data_dir, sheaf_data, update_every_n: int):
        super(SheafDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,sheaf_data), 'rb') as handle:
                self.sheaf_data = torch.FloatTensor(pickle.load(handle)).t()
        else:
            self.sheaf_data = torch.FloatTensor(sheaf_data).t()
        self.update_every_n = update_every_n

    def __len__(self):
        #return self.sheaf_data.size(0)
        # HACKY SOLUTION
        return self.update_every_n

    def __getitem__(self, index):
        return self.sheaf_data[0]
