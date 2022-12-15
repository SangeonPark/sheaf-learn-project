from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#import torch_sparse
class sheaf_learner(object):
    """docstring for sheaf_learner
        rhop: consider r-hop neighbhorhood for pairwise SPSD matrix
    """
    def __init__(self, arg):
        super(sheaf_learner, self).__init__()
        self.arg = arg

    def cholesky_parametrize(self):

        return None

    #def generate_SPSD(self):





#class coboundary(nn.Module):
#    """docstring for coboundary"""
#    def __init__(self, Ne, Nv, de, dv, edges):
#        super(coboundary, self).__init__()
#        self.coboundary_vector = torch.zeros()
#        self.coboundary_matrix = torch.zeros()

#    def forward(self, x):







class coboundary_learner(pl.LightningModule):
    """docstring for coboundary_learner"""
    def __init__(self, graph, Nv, dv, Ne, de, alpha, beta, learning_rate, true_sheaf = None):
        super(coboundary_learner, self).__init__()
        self.save_hyperparameters()
        self.graph = graph
        self.learning_rate = learning_rate
        #self.backbone = MLP(*modelparams)

        self.Ne = Ne
        self.Nv = Nv
        self.de = de
        self.dv = dv
        self.alpha = alpha
        self.beta  = beta

        # Coboundary maps for vertex where edge is outgoing
        self.coboundary_vec_out = nn.ParameterList([nn.Parameter(torch.randn(de, dv)) for i in range(self.Ne)])

        # Coboundary maps for vertex where edge is coming in
        self.coboundary_vec_in  = nn.ParameterList([nn.Parameter(torch.randn(de, dv)) for i in range(self.Ne)])

        # Lifting graph to a sheaf
        #if do_graph_to_sheaf:
        #    # Later change to a more general backbone
        #    self.graph_to_sheaf = nn.Linear(self.input_dim, self.Nv*self.dv)

        for i in range(self.Ne):
            nn.init.normal_(self.coboundary_vec_out[i])
            nn.init.normal_(self.coboundary_vec_in[i])
            #print(self.coboundary_vec_out[i].size())


    def _common_step(self, batch, batch_idx, stage: str):
        #print(batch.size())
        data_covariance = torch.matmul(batch.t(), batch)
        #print(batch.size())
        coboundary = torch.zeros((self.Ne*self.de, self.Nv*self.dv))

        for i, (v1, v2) in enumerate(self.graph.edges):
            coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]

        sheaf_laplacian = torch.matmul(coboundary.t(), coboundary).to(batch)

        # Implement Sparse Matrix Multiplication - torch or torch_sparse
        dirichlet_energy = torch.trace(torch.matmul(data_covariance, sheaf_laplacian))



        reg_connectivity = torch.tensor(0.)
        diagonal_norm = torch.tensor(0.)

        for i in range(self.Nv):
            reg_connectivity = -torch.log(torch.trace(sheaf_laplacian[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv]))
            diagonal_norm = torch.norm(sheaf_laplacian[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv])


        reg_sparsity = torch.norm(sheaf_laplacian) - diagonal_norm
        #print("sparsity: ",reg_sparsity)
        #print("connectivity ", reg_connectivity)
        loss = dirichlet_energy + self.alpha * reg_connectivity + self.beta * reg_sparsity
        return loss, dirichlet_energy, reg_connectivity, reg_sparsity


    def calculate_error(self, true_sheaf):

        return torch.norm(self.build_laplacian_from_vector()-true_sheaf)


    def forward(self):

        return self.build_laplacian_from_vector()





    def build_laplacian_from_vector(self):

        coboundary = torch.zeros((self.Ne*self.de, self.Nv*self.dv))

        for i, (v1, v2) in enumerate(self.graph.edges):
            coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]

        sheaf_laplacian = torch.matmul(coboundary.t(), coboundary)

        return sheaf_laplacian

    #def dirichlet_energy(self, x):

    #def regularizer_sparsity(self, x):

    #def regularizer_connectivity(self, x):
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        loss, dirichlet_energy, reg_connectivity, reg_sparsity = self._common_step(batch, batch_idx, "train")
        self.log_dict({"train_loss": loss, "train_dirichlet":dirichlet_energy, "train_conn":reg_connectivity, "train_sparse":reg_sparsity})
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        loss, dirichlet_energy, reg_connectivity, reg_sparsity = self._common_step(batch, batch_idx, "val")
        self.log_dict({"val_loss": loss, "val_dirichlet":dirichlet_energy, "val_conn":reg_connectivity, "val_sparse":reg_sparsity})
        return loss

    def test_step(self, batch, batch_idx):

        loss, dirichlet_energy, reg_connectivity, reg_sparsity = self._common_step(batch, batch_idx, "test")
        return loss

    def predict_step(self):
        return self.build_laplacian_from_vector()


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class coboundary_learner_on_graph_signal(coboundary_learner):
    """docstring for coboundary_learner_on_graph"""
    def __init__(self, graph, Nv, dv, Ne, de, alpha, beta, learning_rate, input_dim, channels, input_dropout, dropout, true_sheaf = None):
        super(coboundary_learner_on_graph_signal, self).__init__(graph, Nv, dv, Ne, de, alpha, beta, learning_rate, true_sheaf = None)
        # Lifting graph to a sheaf
        # Later change to a more general backbone
        # Add Hidden Channel

        self.graph_to_sheaf = nn.Linear(input_dim, self.dv*channels)
        self.graph_to_sheaf2 = nn.Linear(self.dv*channels, self.dv*channels)
        self.input_dropout  = input_dropout
        self.dropout        = dropout


    def _common_step(self, batch, batch_idx, stage: str):
        batch, _  = batch
        #print(batch.size())
        training = False
        if stage == 'train' or 'fit':
            training = True
        batch = F.dropout(batch, p = self.input_dropout, training=training)
        batch = self.graph_to_sheaf(batch)
        batch = F.elu(batch)
        batch = F.dropout(batch, p = self.dropout, training=training)
        batch = self.graph_to_sheaf2(batch)
        #if self.use_act:
        #    x = F.elu(x)
        batch = batch.view(-1, self.Nv * self.dv)


        data_covariance = torch.matmul(batch.t(), batch)
        #print(batch.size())
        coboundary = torch.zeros((self.Ne*self.de, self.Nv*self.dv))

        for i, (v1, v2) in enumerate(self.graph.edges):
            coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]

        sheaf_laplacian = torch.matmul(coboundary.t(), coboundary).to(batch)

        # Implement Sparse Matrix Multiplication - torch or torch_sparse
        dirichlet_energy = torch.trace(torch.matmul(data_covariance, sheaf_laplacian))



        reg_connectivity = torch.tensor(0.)
        diagonal_norm = torch.tensor(0.)

        for i in range(self.Nv):
            reg_connectivity = -torch.log(torch.trace(sheaf_laplacian[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv]))
            diagonal_norm = torch.norm(sheaf_laplacian[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv])


        reg_sparsity = torch.norm(sheaf_laplacian) - diagonal_norm
        #print("sparsity: ",reg_sparsity)
        #print("connectivity ", reg_connectivity)
        loss = dirichlet_energy + self.alpha * reg_connectivity + self.beta * reg_sparsity
        return loss, dirichlet_energy, reg_connectivity, reg_sparsity

