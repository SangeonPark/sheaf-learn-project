from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch_sparse
import torch.nn.functional as F

# Quick and dirty solution, refactor later

class sheaf_diffusion(pl.LightningModule):
    """docstring for sheaf_diffusion"""
    def __init__(self, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, sheaf_laplacian, mask, use_act, dropout, learning_rate = 0.01):
        super(sheaf_diffusion, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate


        self.layers = layers
        self.hidden_channels = channels
        self.hidden_dim = channels * dv
        self.graph_size = Nv
        self.final_d = dv
        self.register_buffer("sheaf_laplacian", sheaf_laplacian)
        # Copy weights from sheaf(coboundary) learner
        self.graph_to_sheaf = nn.Linear(input_dim, dv*channels)
        self.graph_to_sheaf2 = nn.Linear(dv*channels, dv*channels)
        self.mask = mask
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.dropout = dropout
        self.use_act = use_act





        self.lin_left_weights  = nn.ModuleList()
        self.lin_right_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        #self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        #if self.second_linear:
        #    self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)



    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x


    def forward(self, x):
        return self._common_step(x, None)


    def _common_step(self, batch, batch_idx, stage: str):

        #print(batch)
        batch, _  = batch
        #print(batch.size())
        batch = self.graph_to_sheaf(batch)
        batch = F.elu(batch)
        batch = self.graph_to_sheaf2(batch)
        #if self.use_act:
        #    x = F.elu(x)
        #batch = batch.view(-1, self.Nv * self.dv)
        batch = batch.view(self.graph_size * self.final_d, -1)

        #x = F.dropout(x, p=self.input_dropout, training=self.training)
        #x = self.lin1(x)
        #if self.use_act:
        #    x = F.elu(x)
        #x = F.dropout(x, p=self.dropout, training=self.training)

      #  #if self.second_linear:
        #    x = self.lin12(x)
        #x = x.view(self.graph_size * self.final_d, -1)

        x = batch
        x0 = x
        for layer in range(self.layers):
            #if layer == 0 or self.nonlinear:
            #    x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
            #    maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
            #    L, trans_maps = self.laplacian_builder(maps)
            #    self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])

            # Use the adjacency matrix rather than the diagonal
            #x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            #print(x.size())
            #print(self.sheaf_laplacian.size())
            x = torch.matmul(self.sheaf_laplacian, x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)



    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        out = self._common_step(batch, batch_idx, "train")[self.mask['train_mask']]
        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['train_mask']])
        loss = nll
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        out = self._common_step(batch, batch_idx, "val")[self.mask['val_mask']]
        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['val_mask']])
        loss = nll
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        out = self._common_step(batch, batch_idx, "test")[self.mask['test_mask']]
        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['test_mask']])
        loss = nll
        self.log("test_loss", loss)
        return loss



    def predict_step(self):
        return None


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

