import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from torch_geometric.utils import from_networkx, to_networkx


import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger



from typing import Callable, Optional


import sys
sys.path.insert(0,'..')
from utils.heterophilic import get_dataset, get_fixed_splits, WebKB, Actor
from project.sheaf_datamodule import SheafDataset_ForGraphs, SheafDataModule_ForGraphs
from project.sheaf_learner import coboundary_learner_on_graph_signal

from project.sheaf_simultaneous_functor_edgewise  import sheaf_gradient_flow_functor


import wandb

wandb.login()

dataset = get_dataset('texas')
data_source = dataset[0]




sweep_config = {
    'method': 'random'
    }


metric = {
    'name': 'val_acc',
    'goal': 'maximize'
    }
sweep_config['metric'] = metric

parameters_dict = {
    'dv': {
        'distribution': 'categorical',
        'values': [2,3,4,5]
    },
    'de': {
        'distribution': 'categorical',
        'values': [2,3,4,5]
    },
    'input_dropout': {
        'distribution': 'categorical',
        #'values':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        'values':[0.0, 0.1]
    },
    'dropout': {
        'distribution': 'categorical',
        #'values':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        'values':[0.0, 0.1, 0.2]
    },
    'fold': {
        'distribution': 'categorical',
        #'values':[0,1,2,3,4,5,6,7,8,9],
        'values':[0,1,2,3,4,5,6,7,8,9]
    },
    'layers':{
        'distribution': 'categorical',
        'values': [1,2,3]
    },
    'channels':{
        'distribution': 'categorical',
        'values': [2,4,8,16,32]
    },
    'batch_size':{
        'distribution': 'categorical',
        'values': [20,50]
    },
    'first_hidden':{
        'distribution':'categorical',
        'values':[1000, 500, 100]
    },
    'second_hidden':{
        'distribution':'categorical',
        'values':[1000, 500, 100]
    },
    'free_potential':{
        'distribution':'categorical',
        'values':[0,1,2]
    },





}

    #def __init__(self, graph, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, potential, mask, use_act, augmented, add_lp, add_hp, dropout, input_dropout, perturb_diagonal, free_potential, stalk_mixing, channel_mixing, learning_rate = 0.01):



sweep_config['parameters'] = parameters_dict

import pprint
pprint.pprint(sweep_config)
#sweep_id = wandb.sweep(sweep_config, project="Sweep_functor_onlyxflow_radial_Dec_24_texas_fold0")
#/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/sweeps/zvf0s60g
sweep_id = 'saepark/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/zvf0s60g'
print(sweep_id)




def sweep_iteration_1():
    # set up W&B logger
    class PrintCallbacks(Callback):
        def on_init_start(self, trainer):
            print("Starting to init trainer!")

        def on_init_end(self, trainer):
            print("Trainer is init now")

        def on_train_end(self, trainer, pl_module):
            print("Training ended")

    wandb.init()    # required to have access to `wandb.config`
    wandb_logger_diffusion = WandbLogger(project="Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.000, patience=30, verbose=False, mode='min')
    path = f'/root/projects/temp_checkpoints/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/texas_{wandb.config.fold}/dv{wandb.config.dv}-de{wandb.config.dv}-layers{wandb.config.layers}-channels{wandb.config.channels}-firsthidden{wandb.config.first_hidden}-secondhidden{wandb.config.second_hidden}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=path,
        filename="functor_onlyxflow_radial_dec24-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}",
        save_top_k=50,
        mode="max",
    )
    data = get_fixed_splits(data_source, 'texas', wandb.config.fold)
    graph = to_networkx(data)
    mask = {'train_mask':data.train_mask,
        'val_mask':data.val_mask,
        'test_mask':data.test_mask
       }

    # setup data
    sheaf_dm = SheafDataModule_ForGraphs(data, wandb.config.batch_size , 1)

    # setup model - note how we refer to sweep parameters with wandb.config
    model_diffusion = sheaf_gradient_flow_functor(graph, 183, wandb.config.dv, graph.number_of_edges(), wandb.config.de, wandb.config.layers, 1703,5, wandb.config.channels, True, True, 'radial', mask, True,True,True,True, wandb.config.dropout,wandb.config.input_dropout,wandb.config.free_potential, wandb.config.first_hidden, wandb.config.second_hidden, 1e-4)

    #def __init__(self, graph, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, potential, mask, use_act, augmented, add_lp, add_hp, dropout, input_dropout, free_potential, first_hidden, second_hidden, learning_rate = 0.01):

    trainer = Trainer(accelerator='gpu',devices=[1],callbacks=[TQDMProgressBar(refresh_rate=10),PrintCallbacks(),early_stop_callback,checkpoint_callback],logger=wandb_logger_diffusion)
    lr_finder = trainer.tuner.lr_find(model_diffusion, sheaf_dm)
    new_lr = lr_finder.suggestion()
    model_diffusion.hparams.learning_rate = new_lr
    model_diffusion.learning_rate = new_lr
    # train
    trainer.fit(model_diffusion, sheaf_dm)
    import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from torch_geometric.utils import from_networkx, to_networkx


import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger



from typing import Callable, Optional


import sys
sys.path.insert(0,'..')
from utils.heterophilic import get_dataset, get_fixed_splits, WebKB, Actor
from project.sheaf_datamodule import SheafDataset_ForGraphs, SheafDataModule_ForGraphs
from project.sheaf_learner import coboundary_learner_on_graph_signal

from project.sheaf_simultaneous_functor_edgewise  import sheaf_gradient_flow_functor


import wandb

wandb.login()

dataset = get_dataset('texas')
data_source = dataset[0]




sweep_config = {
    'method': 'random'
    }


metric = {
    'name': 'val_acc',
    'goal': 'maximize'
    }
sweep_config['metric'] = metric

parameters_dict = {
    'dv': {
        'distribution': 'categorical',
        'values': [2,3,4,5]
    },
    'de': {
        'distribution': 'categorical',
        'values': [2,3,4,5]
    },
    'input_dropout': {
        'distribution': 'categorical',
        #'values':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        'values':[0.0, 0.1]
    },
    'dropout': {
        'distribution': 'categorical',
        #'values':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        'values':[0.0, 0.1, 0.2]
    },
    'fold': {
        'distribution': 'categorical',
        #'values':[0,1,2,3,4,5,6,7,8,9],
        'values':[0,1,2,3,4,5,6,7,8,9]
    },
    'layers':{
        'distribution': 'categorical',
        'values': [1,2,3]
    },
    'channels':{
        'distribution': 'categorical',
        'values': [2,4,8,16,32]
    },
    'batch_size':{
        'distribution': 'categorical',
        'values': [20,50]
    },
    'first_hidden':{
        'distribution':'categorical',
        'values':[1000, 500, 100]
    },
    'second_hidden':{
        'distribution':'categorical',
        'values':[1000, 500, 100]
    },
    'free_potential':{
        'distribution':'categorical',
        'values':[0,1,2]
    },





}

    #def __init__(self, graph, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, potential, mask, use_act, augmented, add_lp, add_hp, dropout, input_dropout, perturb_diagonal, free_potential, stalk_mixing, channel_mixing, learning_rate = 0.01):



sweep_config['parameters'] = parameters_dict

import pprint
pprint.pprint(sweep_config)
#sweep_id = wandb.sweep(sweep_config, project="Sweep_functor_onlyxflow_radial_Dec_24_texas_fold0")
#/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/sweeps/zvf0s60g
sweep_id = 'saepark/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/zvf0s60g'
print(sweep_id)




def sweep_iteration_0():
    # set up W&B logger
    class PrintCallbacks(Callback):
        def on_init_start(self, trainer):
            print("Starting to init trainer!")

        def on_init_end(self, trainer):
            print("Trainer is init now")

        def on_train_end(self, trainer, pl_module):
            print("Training ended")

    wandb.init()    # required to have access to `wandb.config`
    wandb_logger_diffusion = WandbLogger(project="Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.000, patience=30, verbose=False, mode='min')
    path = f'/root/projects/temp_checkpoints/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/texas_{wandb.config.fold}/dv{wandb.config.dv}-de{wandb.config.dv}-layers{wandb.config.layers}-channels{wandb.config.channels}-firsthidden{wandb.config.first_hidden}-secondhidden{wandb.config.second_hidden}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=path,
        filename="functor_onlyxflow_radial_dec24-{epoch:02d}-{val_loss:.2f}-{val_acc:.4f}",
        save_top_k=50,
        mode="max",
    )
    data = get_fixed_splits(data_source, 'texas', wandb.config.fold)
    graph = to_networkx(data)
    mask = {'train_mask':data.train_mask,
        'val_mask':data.val_mask,
        'test_mask':data.test_mask
       }

    # setup data
    sheaf_dm = SheafDataModule_ForGraphs(data, wandb.config.batch_size, 1)

    # setup model - note how we refer to sweep parameters with wandb.config
    model_diffusion = sheaf_gradient_flow_functor(graph, 183, wandb.config.dv, graph.number_of_edges(), wandb.config.de, wandb.config.layers, 1703,5, wandb.config.channels, True, True, 'radial', mask, True,True,True,True, wandb.config.dropout,wandb.config.input_dropout,wandb.config.free_potential, wandb.config.first_hidden, wandb.config.second_hidden, 1e-4)

    #def __init__(self, graph, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, potential, mask, use_act, augmented, add_lp, add_hp, dropout, input_dropout, free_potential, first_hidden, second_hidden, learning_rate = 0.01):

    trainer = Trainer(accelerator='gpu',devices=[0],callbacks=[TQDMProgressBar(refresh_rate=10),PrintCallbacks(),early_stop_callback,checkpoint_callback],logger=wandb_logger_diffusion)
    lr_finder = trainer.tuner.lr_find(model_diffusion, sheaf_dm)
    new_lr = lr_finder.suggestion()
    model_diffusion.hparams.learning_rate = new_lr
    model_diffusion.learning_rate = new_lr
    # train
    trainer.fit(model_diffusion, sheaf_dm)


    from os import listdir
    from os.path import isfile, join
    mypath = f'/root/projects/temp_checkpoints/Sweep_functor_onlyxflow_radial_Dec_24_texas_fold_all/texas_{wandb.config.fold}/dv{wandb.config.dv}-de{wandb.config.dv}-layers{wandb.config.layers}-channels{wandb.config.channels}-firsthidden{wandb.config.first_hidden}-secondhidden{wandb.config.second_hidden}'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for files in onlyfiles:
        path = os.path.join(mypath,files)
        model_diffusion = sheaf_gradient_flow_functor.load_from_checkpoint(path)
        trainer.test(model_diffusion, sheaf_dm)






wandb.agent(sweep_id, function=sweep_iteration_0)








wandb.agent(sweep_id, function=sweep_iteration_1)

