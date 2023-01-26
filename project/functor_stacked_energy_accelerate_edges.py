from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import torch.nn.utils.parametrize as P
import networkx as nx

# Quick and dirty solution, refactor later

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu()+X.triu(1).T

    def right_inverse(self, A):
        return A.triu()

def symmetrize(module, name='weight'):
    weight = getattr(module, name, None)
    P.register_parametrization(module, name, Symmetric())
    return module
# Make Linear Backbone  More complicated
# For the Functor Case, Yes

class sheaf_gradient_flow_functor(pl.LightningModule):
    """docstring for sheaf_diffusion"""
    def __init__(self, graph, from_vertex_outgoing_edges, from_vertex_incoming_edges, from_vertex_outgoing_vertex, from_vertex_incoming_vertex, degree_list, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, potential, mask, use_act, augmented, add_lp, add_hp, dropout, input_dropout, free_potential, first_hidden, second_hidden, NVertEnergy, NEdgeEnergy,nonlinearoutput, add_adjacency, add_stalk_mixing, add_channel_mixing, apply_spectralnorm, apply_symmetrize, vertex_potential_type, learning_rate = 1e-4, weight_decay = 5e-4):
        super(sheaf_gradient_flow_functor, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.free_potential = free_potential
        self.graph  = graph
        self.graph_size = len(graph)
        self.layers = layers
        self.hidden_channels = channels
        self.nonlinearoutput = nonlinearoutput
        self.add_adjacency = add_adjacency
        self.add_stalk_mixing= add_stalk_mixing
        self.add_channel_mixing = add_channel_mixing
        self.apply_spectralnorm = apply_spectralnorm
        self.apply_symmetrize = apply_symmetrize
        self.from_vertex_incoming_edges = from_vertex_incoming_edges
        self.from_vertex_incoming_vertex = from_vertex_incoming_vertex
        self.from_vertex_outgoing_edges = from_vertex_outgoing_edges
        self.from_vertex_outgoing_vertex = from_vertex_outgoing_vertex
        self.degree_list = degree_list


        self.graph_size = Nv
        self.final_d = dv
        if add_lp:
            self.final_d += 1
        if add_hp:
            self.final_d += 1

        self.hidden_dim = channels * self.final_d
        self.Ne = Ne
        self.dv = dv
        self.de = de
        self.Nv = Nv

        # MUST ADD
        self.NVertEnergy = NVertEnergy
        self.NEdgeEnergy = NEdgeEnergy

        self.mask = mask
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act
        self.vertex_potential_type = vertex_potential_type

        # ADD NEW OPTIONS
        self.augmented = augmented
        self.add_lp    = add_lp
        self.add_hp    = add_hp
        # Linear Matrics
        # Sheafification
        #self.max_deg = max_deg
        #self.adjacency = torch.tensor(adjacency, requires_grad=False)

        #compress = 100
        #self.initial1 = nn.Linear(input_dim+self.Nv, 500)
        #self.initial2 = nn.Linear(500, 500)
        #self.initial3 = nn.Linear(500, 100)
        #self.initial4 = nn.Linear(300, 100)
        #self.initial5 = nn.Linear(100, 50)
        #self.initial6 = nn.Linear(50, 10)

        #self.initial1 = nn.Linear(input_dim, 183)
        #self.initial2 = nn.Linear(100, 100)
        #self.initial3 = nn.Linear(100, compress)

        # Sheafifiers
        self.graph_to_sheaf = nn.Linear(input_dim * 2, first_hidden)
        self.graph_to_sheaf2  = nn.Linear(first_hidden, second_hidden)
        self.graph_to_sheaf3  = nn.Linear(second_hidden, 2*(self.final_d*self.hidden_channels)+2*dv*de)

        # Mixing Matrices
        if self.add_stalk_mixing:
            if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                self.self_energy_stalk_mixing = nn.Linear(self.final_d, self.final_d, bias=False)
                nn.init.orthogonal_(self.self_energy_stalk_mixing.weight.data)
            elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                self.self_energy_stalk_mixing = spectral_norm(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.orthogonal_(self.self_energy_stalk_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                self.self_energy_stalk_mixing = symmetrize(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.orthogonal_(self.self_energy_stalk_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                self.self_energy_stalk_mixing = spectral_norm(symmetrize(nn.Linear(self.final_d, self.final_d, bias=False)))
                nn.init.orthogonal_(self.self_energy_stalk_mixing.weight.data)

        if self.add_channel_mixing:
            if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                self.self_energy_channel_mixing = nn.Linear(channels, channels, bias=False)
                nn.init.orthogonal_(self.self_energy_channel_mixing.weight.data)
            elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                self.self_energy_channel_mixing = spectral_norm(nn.Linear(channels, channels, bias=False))
                nn.init.orthogonal_(self.self_energy_channel_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                self.self_energy_channel_mixing = symmetrize(nn.Linear(channels, channels, bias=False))
                nn.init.orthogonal_(self.self_energy_channel_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                self.self_energy_channel_mixing = spectral_norm(symmetrize(nn.Linear(channels, channels, bias=False)))
                nn.init.orthogonal_(self.self_energy_channel_mixing.weight.data)


        if self.add_adjacency and self.right_weights:
            if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                self.adjacent_node_energy_channel_mixing = nn.Linear(channels, channels, bias=False)
                nn.init.orthogonal_(self.adjacent_node_energy_channel_mixing.weight.data)
            elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                self.adjacent_node_energy_channel_mixing = spectral_norm(nn.Linear(channels, channels, bias=False))
                nn.init.orthogonal_(self.adjacent_node_energy_channel_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                self.adjacent_node_energy_channel_mixing = symmetrize(nn.Linear(channels, channels, bias=False))
                nn.init.orthogonal_(self.adjacent_node_energy_channel_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                self.adjacent_node_energy_channel_mixing = spectral_norm(symmetrize(nn.Linear(channels, channels, bias=False)))
                nn.init.orthogonal_(self.adjacent_node_energy_channel_mixing.weight.data)


        if self.add_adjacency and self.left_weights:
            if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                self.adjacent_node_energy_stalk_mixing = nn.Linear(self.final_d, self.final_d, bias=False)
                nn.init.orthogonal_(self.adjacent_node_energy_stalk_mixing.weight.data)
            elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                self.adjacent_node_energy_stalk_mixing = spectral_norm(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.orthogonal_(self.adjacent_node_energy_stalk_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                self.adjacent_node_energy_stalk_mixing = symmetrize(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.orthogonal_(self.adjacent_node_energy_stalk_mixing.weight.data)
            elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                self.adjacent_node_energy_stalk_mixing = spectral_norm(symmetrize(nn.Linear(self.final_d, self.final_d, bias=False)))
                nn.init.orthogonal_(self.adjacent_node_energy_stalk_mixing.weight.data)


        self.edge_lin_left_weights  = nn.ModuleList()
        self.edge_lin_right_weights = nn.ModuleList()
        self.vertex_lin_left_weights  = nn.ModuleList()
        self.vertex_lin_right_weights = nn.ModuleList()
        A = nx.adjacency_matrix(graph)
        A = A.toarray()
        L = nx.laplacian_matrix(graph)
        L = L.toarray()
        D = L + A
        A = torch.tensor(A, requires_grad = False)
        D = torch.tensor(D, requires_grad = False)
        D_tilde = torch.diagonal(D) + torch.ones(len(graph))
        D_tilde_inv_sqrt_flat = 1./(torch.sqrt(D_tilde))
        D_tilde_inv_sqrt = torch.diag(D_tilde_inv_sqrt_flat)
        A_tilde = A + torch.eye(len(graph))
        Norm_Lap = torch.eye(len(graph)) - D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        #print(Norm_Lap, Norm_Lap.size())
        self.Norm_Lap = Norm_Lap
        # Edge potentials
        self.potential = nn.ParameterList()
        self.vertex_potential = nn.ParameterList()
        for i in range(self.NEdgeEnergy):
            if self.free_potential == 0 or 1:
                self.potential.append(nn.Parameter(torch.zeros(self.Ne), requires_grad=True))
            if self.free_potential == 2:
                self.potential.append(nn.Parameter(torch.ones(self.Ne), requires_grad=True))
        for i in range(self.NVertEnergy):
            if self.vertex_potential_type == 0:
                self.vertex_potential.append(nn.Parameter(torch.zeros((self.Nv,1)), requires_grad=True))
            elif self.vertex_potential_type == 1:
                self.vertex_potential.append(nn.Parameter(torch.ones((self.Nv,1)), requires_grad=True))

        #self.potential = nn.Parameter(torch.randn(self.Ne), requires_grad=True)
        #if self.free_potential == 0 or 1:
        #    self.potential = nn.Parameter(torch.zeros(self.Ne), requires_grad=True)
        #if self.free_potential == 2:
        #    self.potential = nn.Parameter(torch.ones(self.Ne), requires_grad=True)

        #self.vertex_potential = nn.Parameter(torch.ones((self.Nv,1)), requires_grad=True)
        #self.batch_norms = nn.ModuleList()


        if self.right_weights:
            for i in range(self.layers):
                if self.NEdgeEnergy > 0:
                    self.edge_lin_right_weights.append(nn.ModuleList())
                    for j in range(self.NEdgeEnergy):
                        if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                            self.edge_lin_right_weights[-1].append(nn.Linear(channels, channels, bias=False))
                            nn.init.orthogonal_(self.edge_lin_right_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                            self.edge_lin_right_weights[-1].append(spectral_norm(nn.Linear(channels, channels, bias=False)))
                            nn.init.orthogonal_(self.edge_lin_right_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                            self.edge_lin_right_weights[-1].append(symmetrize(nn.Linear(channels, channels, bias=False)))
                            nn.init.orthogonal_(self.edge_lin_right_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                            self.edge_lin_right_weights[-1].append(spectral_norm(symmetrize(nn.Linear(channels, channels, bias=False))))
                            nn.init.orthogonal_(self.edge_lin_right_weights[-1][-1].weight.data)


                if self.NVertEnergy > 0:
                    self.vertex_lin_right_weights.append(nn.ModuleList())
                    for j in range(self.NVertEnergy):
                        if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                            self.vertex_lin_right_weights[-1].append(nn.Linear(channels, channels, bias=False))
                            nn.init.orthogonal_(self.vertex_lin_right_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                            self.vertex_lin_right_weights[-1].append(spectral_norm(nn.Linear(channels, channels, bias=False)))
                            nn.init.orthogonal_(self.vertex_lin_right_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                            self.vertex_lin_right_weights[-1].append(symmetrize(nn.Linear(channels, channels, bias=False)))
                            nn.init.orthogonal_(self.vertex_lin_right_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                            self.vertex_lin_right_weights[-1].append(spectral_norm(symmetrize(nn.Linear(channels, channels, bias=False))))
                            nn.init.orthogonal_(self.vertex_lin_right_weights[-1][-1].weight.data)




        if self.left_weights:
            for i in range(self.layers):
                if self.NEdgeEnergy > 0:
                    self.edge_lin_left_weights.append(nn.ModuleList())
                    for j in range(self.NEdgeEnergy):
                        if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                            self.edge_lin_left_weights[-1].append(nn.Linear(self.final_d, self.final_d, bias=False))
                            nn.init.orthogonal_(self.edge_lin_left_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                            self.edge_lin_left_weights[-1].append(spectral_norm(nn.Linear(self.final_d, self.final_d, bias=False)))
                            nn.init.orthogonal_(self.edge_lin_left_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                            self.edge_lin_left_weights[-1].append(symmetrize(nn.Linear(self.final_d, self.final_d, bias=False)))
                            nn.init.orthogonal_(self.edge_lin_left_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                            self.edge_lin_left_weights[-1].append(spectral_norm(symmetrize(nn.Linear(self.final_d, self.final_d, bias=False))))
                            nn.init.orthogonal_(self.edge_lin_left_weights[-1][-1].weight.data)

                if self.NVertEnergy > 0:
                    self.vertex_lin_left_weights.append(nn.ModuleList())
                    for j in range(self.NVertEnergy):
                        if self.apply_symmetrize == False and self.apply_spectralnorm == False:
                            self.vertex_lin_left_weights[-1].append(nn.Linear(self.final_d, self.final_d, bias=False))
                            nn.init.orthogonal_(self.vertex_lin_left_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == False and self.apply_spectralnorm==True:
                            self.vertex_lin_left_weights[-1].append(spectral_norm(nn.Linear(self.final_d, self.final_d, bias=False)))
                            nn.init.orthogonal_(self.vertex_lin_left_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==False:
                            self.vertex_lin_left_weights[-1].append(symmetrize(nn.Linear(self.final_d, self.final_d, bias=False)))
                            nn.init.orthogonal_(self.vertex_lin_left_weights[-1][-1].weight.data)
                        elif self.apply_symmetrize == True and self.apply_spectralnorm==True:
                            self.vertex_lin_left_weights[-1].append(spectral_norm(symmetrize(nn.Linear(self.final_d, self.final_d, bias=False))))
                            nn.init.orthogonal_(self.vertex_lin_left_weights[-1][-1].weight.data)




        # Weights for the update step
        self.epsilons = nn.ParameterList()

        self.relative_contribution_of_energies = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))
            self.relative_contribution_of_energies.append(nn.Parameter(torch.zeros(self.NEdgeEnergy+self.NVertEnergy+2+1),requires_grad=True))


        #self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        #if self.second_linear:
        #    self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.out_layer1 = nn.Linear(self.hidden_dim, self.hidden_dim*2)
        self.out_layer2 = nn.Linear(self.hidden_dim*2, self.hidden_dim*2)
        self.out_final = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.maxa = 0


    def augment(self, matrix):
        if self.add_lp or self.add_hp:
            if self.add_lp and self.add_hp:
                new_matrix = torch.zeros((self.dv+2, self.dv+2))

                new_matrix[0:self.dv , 0:self.dv] = matrix
                new_matrix[self.dv , self.dv ] = 1.
                new_matrix[self.dv+1 , self.dv+1 ] = -1.

            elif self.add_lp:
                new_matrix = torch.zeros((self.dv+1, self.dv+1))

                new_matrix[0:self.dv , 0:self.dv] = matrix
                new_matrix[self.dv , self.dv ] = -1.


            elif self.add_hp:
                new_matrix = torch.zeros((self.dv+1, self.dv+1))
                new_matrix[0:self.dv , 0:self.dv] = matrix
                new_matrix[self.dv , self.dv ] = 1.

        return new_matrix.to(matrix)


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

    def get_D_matrix(self, matrix_list, add_diagonal):
        new_matrix_list = matrix_list + add_diagonal
        u, s, _ = torch.svd(new_matrix_list)
        vals = s
        vecs = u
        good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
        vals = torch.abs(vals).pow(-0.5).where(good, torch.zeros((), device=matrix_list.device, dtype=matrix_list.dtype))
        D_pow = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)
        return D_pow

    def normalize_laplacian(self, laplacian, add_diagonal):
        diagonal_blocks = torch.FloatTensor(self.Nv, self.dv, self.dv).to(laplacian)
        for i in range(self.Nv):
            diagonal_blocks[i,0*self.dv:(1)*self.dv, 0*self.dv:(1)*self.dv] = laplacian[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv] + add_diagonal[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv]

        u, s, _ = torch.svd(diagonal_blocks)
        vals = torch.flatten(s)
        vecs = torch.block_diag(*u)
        #diagonal_blocks = [laplacian[i*self.dv:(i+1)*self.dv, i*self.dv:(i+1)*self.dv] for i in range(self.Nv)]
        #D = torch.block_diag(*diagonal_blocks)
        #D = D + add_diagonal
        #vecs, vals, _ = torch.linalg.svd(D)

        #sorted, _ = torch.sort(vals)
        #print("low: ",sorted[:10])
        #print("high: ",sorted[-10:])
        good = vals > vals.max(-1, True).values * vals.size(-1) * torch.finfo(vals.dtype).eps
        vals = torch.abs(vals).pow(-0.5).where(good, torch.zeros((), device=laplacian.device, dtype=laplacian.dtype))
        D_pow = (vecs * vals.unsqueeze(-2)) @ torch.transpose(vecs, -2, -1)

        #evals, evecs = torch.linalg.eig(D)
        #evpow = evals**(-1/2)
        #D_pow = torch.matmul (evecs, torch.matmul (torch.diag (evpow), torch.inverse (evecs))).real

        normalized_laplacian = D_pow* laplacian * D_pow
        return normalized_laplacian


    def _common_step(self, batch, batch_idx, stage: str):

        #print(batch)
        if stage == 'train':
            training = True
        else:
            training = False
        batch, _  = batch
        self.Norm_Lap = self.Norm_Lap.to(batch)
        self.Norm_Lap.requires_grad_ = False
        batch = F.dropout(batch, p=self.input_dropout, training=training)
        batch = batch.reshape(self.Nv,-1)

        batch_vertex = torch.zeros((self.Nv, self.final_d*self.hidden_channels)).to(batch)
        restriction = torch.zeros((self.Ne, 2, self.de, self.dv)).to(batch)
        diagonal_normalizer = torch.zeros((self.NEdgeEnergy, self.Nv, self.dv, self.dv)).to(batch)

        potential_multiplier = torch.zeros((self.NEdgeEnergy, self.Ne, self.dv, self.dv)).to(batch)
        for j in range(self.NEdgeEnergy):
            if self.free_potential == 0:
                potential_multiplier[j] = (1.+torch.tanh(self.potential[j])).unsqueeze(1).unsqueeze(2).repeat(1, self.dv, self.dv).to(batch)
            if self.free_potential == 1:
                potential_multiplier[j] = (1.+self.potential[j]).unsqueeze(1).unsqueeze(2).repeat(1, self.dv, self.dv).to(batch)

            if self.free_potential == 2:
                potential_multiplier[j] = self.potential[j].unsqueeze(1).unsqueeze(2).repeat(1, self.dv, self.dv).to(batch)



        for i, (v1, v2) in enumerate(self.graph.edges):

            edge = self.graph_to_sheaf(torch.cat((batch[v1], batch[v2])))
            edge = F.elu(edge)
            edge = F.dropout(edge, p=self.dropout, training=training)
            edge = self.graph_to_sheaf2(edge)
            edge = F.elu(edge)
            edge = F.dropout(edge, p=self.dropout, training=training)
            edge = self.graph_to_sheaf3(edge)

            batch_vertex[v1] = batch_vertex[v1] + edge[:self.final_d*self.hidden_channels]
            batch_vertex[v2] = batch_vertex[v2] + edge[self.final_d*self.hidden_channels:2*self.final_d*self.hidden_channels]

            restriction[i] = edge[2*self.final_d*self.hidden_channels:].reshape(2,self.de,self.dv)


        # think about using edge map directly
        self.degree_list = self.degree_list.to(batch)
        batch_vertex = batch_vertex/self.degree_list.unsqueeze(1)



        self_11_template = torch.matmul(torch.transpose(restriction[:,0,:,:],1,2),restriction[:,0,:,:])
        self_22_template = torch.matmul(torch.transpose(restriction[:,1,:,:],1,2),restriction[:,1,:,:])
        cross_12_template = -1 * torch.matmul(torch.transpose(restriction[:,0,:,:],1,2),restriction[:,1,:,:])
        cross_21_template = -1 * torch.matmul(torch.transpose(restriction[:,1,:,:],1,2),restriction[:,0,:,:])

        self_11_template = self_11_template.unsqueeze(0).repeat(self.NEdgeEnergy, 1, 1, 1)
        self_22_template = self_22_template.unsqueeze(0).repeat(self.NEdgeEnergy, 1, 1, 1)
        cross_12_template = cross_12_template.unsqueeze(0).repeat(self.NEdgeEnergy, 1, 1, 1)
        cross_21_template = cross_21_template.unsqueeze(0).repeat(self.NEdgeEnergy, 1, 1, 1)

        self_11 = torch.zeros((self.NEdgeEnergy, self.Ne, self.dv, self.dv)).to(batch)
        self_22 = torch.zeros((self.NEdgeEnergy, self.Ne, self.dv, self.dv)).to(batch)
        cross_12 = torch.zeros((self.NEdgeEnergy, self.Ne, self.dv, self.dv)).to(batch)
        cross_21 = torch.zeros((self.NEdgeEnergy, self.Ne, self.dv, self.dv)).to(batch)
        for j in range(self.NEdgeEnergy):
            self_11[j] = potential_multiplier[j] * self_11_template[j]
            self_22[j] = potential_multiplier[j] * self_22_template[j]
            cross_12[j] = potential_multiplier[j] * cross_12_template[j]
            cross_21[j] = potential_multiplier[j] * cross_21_template[j]

        #self_11 = self_11.unsqueeze(0).repeat(self.Nv, 1, 1, 1)
        #self_22 = self_22.unsqueeze(0).repeat(self.Nv, 1, 1, 1)


        # Think about getting rid of node loop

        for i in range(self.Nv):
            self.from_vertex_outgoing_edges[i] = self.from_vertex_outgoing_edges[i].to(batch).type(torch.int64)
            self.from_vertex_incoming_edges[i] = self.from_vertex_incoming_edges[i].to(batch).type(torch.int64)
            dummy1 = self.from_vertex_outgoing_edges[i].unsqueeze(1).unsqueeze(2).expand(self.from_vertex_outgoing_edges[i].size(0), self.dv, self.dv)
            dummy2 = self.from_vertex_incoming_edges[i].unsqueeze(1).unsqueeze(2).expand(self.from_vertex_incoming_edges[i].size(0), self.dv, self.dv)
            for j in range(self.NEdgeEnergy):
                diagonal_normalizer[j][i] = torch.sum(self_11[j].gather(0, dummy1), 0) + torch.sum(self_22[j].gather(0, dummy2), 0)






        #Averaging
        if stage == 'train':
                eps = torch.zeros(self.dv).uniform_(-0.0001, 0.0001).to(batch)
                eps.requires_grad_ = False
                lap_add_diagonal = torch.diag(1.+eps).to(batch)
                #lap_add_diagonal = torch.eye(self.dv * self.Nv).to(x_diffusion)

        else:
            lap_add_diagonal = torch.eye(self.dv).to(batch)

        D_pow_list = []
        for j in range(self.NEdgeEnergy):
            D_pow_list.append(self.get_D_matrix(diagonal_normalizer[j], lap_add_diagonal))

        x = batch_vertex.reshape(self.graph_size * self.final_d, -1)

        for layer in range(self.layers):

            #x = F.dropout(x, p=self.dropout, training=training)
            x = F.dropout(x, p=self.dropout, training=training)


            #For x1
            if self.add_stalk_mixing:
                x_stalk_mixing   =  self.self_energy_stalk_mixing(x.reshape(-1, self.final_d)).reshape(-1, self.hidden_channels).to(x)
            if self.add_channel_mixing:
                x_channel_mixing =  self.self_energy_channel_mixing(x.reshape(-1, self.hidden_channels)).to(x)

            if self.add_adjacency:
                x_adjacency_update = self.left_right_linear(x, self.adjacent_node_energy_stalk_mixing, self.adjacent_node_energy_channel_mixing)
                x_adjacency_update = x_adjacency_update.t().reshape(-1, self.graph_size).t()
                x_adjacency_update = (self.Norm_Lap @ x_adjacency_update).t()
                x_adjacency_update = x_adjacency_update.reshape(-1, self.graph_size * self.final_d).t()



            x_vertex_update = torch.zeros(self.NVertEnergy, self.graph_size*self.final_d, self.hidden_channels).to(x)
            for j in range(self.NVertEnergy):
                x_vertex_feats = self.left_right_linear(x, self.vertex_lin_left_weights[layer][j], self.vertex_lin_right_weights[layer][j])
                #final try
                maxelement = torch.max(torch.abs(self.vertex_potential[j]))
                #x_vertex_update[j] = (self.vertex_potential[j]/maxelement).tile(self.final_d,1).reshape(self.graph_size*self.final_d, 1) * x_vertex_feats
                if self.vertex_potential_type == 1:
                    x_vertex_update[j] = (self.vertex_potential[j]/maxelement).tile(self.final_d,1).reshape(self.graph_size*self.final_d, 1) * x_vertex_feats
                if self.vertex_potential_type == 0:
                    x_vertex_update[j] = (1+torch.tanh(self.vertex_potential[j])).tile(self.final_d,1).reshape(self.graph_size*self.final_d, 1) * x_vertex_feats

            x_edge_update = torch.zeros(self.NEdgeEnergy, self.graph_size*self.final_d, self.hidden_channels).to(x)

            for j in range(self.NEdgeEnergy):
                x_diffusion = self.left_right_linear(x, self.edge_lin_left_weights[layer][j], self.edge_lin_right_weights[layer][j])
                x_diffusion = x_diffusion.reshape(self.graph_size, self.final_d, self.hidden_channels)
                x_diffusion_update = torch.zeros((self.Nv, self.final_d, self.hidden_channels)).to(batch)
                for i in range(self.Nv):
                    selfsum = torch.matmul( torch.matmul(torch.matmul(D_pow_list[j][i], diagonal_normalizer[j][i]), D_pow_list[j][i])  , x_diffusion[i][:-2] )
                    selfsum = F.pad(selfsum,(0,0,0,2))
                    x_diffusion_update[i] += selfsum
                    if(len(self.from_vertex_outgoing_vertex[i])>0):
                        dummy1 = self.from_vertex_outgoing_edges[i].unsqueeze(1).unsqueeze(2).expand(self.from_vertex_outgoing_edges[i].size(0), self.dv, self.dv)
                        x_otherside_outgoingedge = x_diffusion[self.from_vertex_outgoing_vertex[i]]

                        crosssum_outgoingedges = torch.matmul(F.pad(torch.matmul(torch.matmul(D_pow_list[j][i].unsqueeze(0).repeat(len(self.from_vertex_outgoing_vertex[i]),1,1), cross_12[j].gather(0, dummy1)), D_pow_list[j][self.from_vertex_outgoing_vertex[i]]),(0,2,0,2)),x_otherside_outgoingedge)
                        x_diffusion_update[i] += torch.sum(crosssum_outgoingedges, dim=0)


                    if(len(self.from_vertex_incoming_vertex[i])>0):
                        x_otherside_incomingedge = x_diffusion[self.from_vertex_incoming_vertex[i]]
                        dummy2 = self.from_vertex_incoming_edges[i].unsqueeze(1).unsqueeze(2).expand(self.from_vertex_incoming_edges[i].size(0), self.dv, self.dv)
                        crosssum_incomingedges = torch.matmul(F.pad(torch.matmul(torch.matmul(D_pow_list[j][i].unsqueeze(0).repeat(len(self.from_vertex_incoming_vertex[i]),1,1), cross_21[j].gather(0, dummy2)), D_pow_list[j][self.from_vertex_incoming_vertex[i]]),(0,2,0,2)),x_otherside_incomingedge)
                        x_diffusion_update[i] += torch.sum(crosssum_incomingedges, dim=0)


                x_edge_update[j] = x_diffusion_update.reshape(self.graph_size * self.final_d, -1)

            #x_diffusion_update = 0.5 * x_diffusion_update
            #x_vertex_update = 0.5 * x_vertex_update
            #x_update = 0.25*(x_diffusion_update + x_vertex_update + x_stalk_mixing + x_channel_mixing)
            #x_update = (1./(torch.abs(self.relative_weights[layer][0])+torch.abs(self.relative_weights[layer][1])))*(self.relative_weights[layer][0]*x_diffusion_update + self.relative_weights[layer][1]*x_vertex_update)
            #x_update = x_diffusion_update + x_vertex_update
            #x_update = torch.zeros(self.graph_size * self.final_d, self.hidden_channels).to(x)
            x_vertex_update_sum = torch.zeros(self.graph_size * self.final_d, self.hidden_channels).to(x)
            x_edge_update_sum = torch.zeros(self.graph_size * self.final_d, self.hidden_channels).to(x)
            x_update_sum = torch.zeros(self.graph_size * self.final_d, self.hidden_channels).to(x)
            #x_vertex_update = torch.zeros(self.graph_size * self.final_d, self.hidden_channels).to(x)
            #edge_normalizer = torch.zeros(1).to(x)
            #vertex_normalizer = torch.zeros(1).to(x)
            pos_restricted_weighting = 1+torch.tanh(self.relative_contribution_of_energies[layer])

            #normalizer = torch.sum(pos_restricted_weighting)
            for j in range(self.NEdgeEnergy):
                #x_update = x_update + (1+torch.tanh(self.edge_relative_weights[layer][j])) * x_edge_update[j]
                x_edge_update_sum = x_edge_update_sum + (1+torch.tanh(pos_restricted_weighting[j])) * x_edge_update[j]
                #edge_normalizer = edge_normalizer + 2*(1+torch.tanh(self.edge_relative_weights[layer][j]))

            for j in range(self.NVertEnergy):
                #x_update = x_update + (1+torch.tanh(self.vertex_relative_weights[layer][j])) * x_vertex_update[j]
                x_vertex_update_sum = x_vertex_update_sum + (1+torch.tanh(pos_restricted_weighting[j+self.NEdgeEnergy])) * x_vertex_update[j]
                #vertex_normalizer = vertex_normalizer + (1+torch.tanh(self.vertex_relative_weights[layer][j]))
            #x_update = (torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x_update
            #x_update = x_update + x_adjacency_update
            #x_update = (1./(normalizer+2))*x_update




            normalizer = torch.zeros(1).to(x)
            if self.NEdgeEnergy > 0:
                normalizer += torch.sum(pos_restricted_weighting[:self.NEdgeEnergy])
                #x_edge_update_sum *= 0.5
                #x_edge_update_sum *= 1
                x_update_sum += x_edge_update_sum
            if self.NVertEnergy > 0:
                normalizer += torch.sum(pos_restricted_weighting[self.NEdgeEnergy:self.NEdgeEnergy+self.NVertEnergy])
                x_update_sum += x_vertex_update_sum
            if self.add_adjacency:
                normalizer += pos_restricted_weighting[-3]
                x_adjacency_update *= 0.5
                x_adjacency_update *= pos_restricted_weighting[-3]
                x_update_sum += x_adjacency_update
            if self.add_stalk_mixing:
                normalizer += pos_restricted_weighting[-2]
                x_stalk_mixing *= pos_restricted_weighting[-2]
                x_update_sum += x_stalk_mixing
            if self.add_channel_mixing:
                normalizer += pos_restricted_weighting[-1]
                x_channel_mixing *= pos_restricted_weighting[-1]
                x_update_sum += x_channel_mixing




            #final_out_sum = torch.sum(pos_restricted)
            x_update = (1./normalizer)*(x_update_sum)



            if self.use_act:
                x_update = F.elu(x_update)
                #x1_update = F.elu(x1_update)
                #x2_update = F.elu(x2_update)

            #print("x0 size:",x0.size())
            #print("x_update size:", x_update.size())
            #x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x_update
            #x = x0
            #x = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x - x_update
            x = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x - x_update

            #x0 = x0 - x0_update
            #x1 = x1 - x1_update
            #x2 = x2 - x2_update

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        x = self.out_layer1(x)
        if self.nonlinearoutput:
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=training)
            x = self.out_layer2(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=training)
        x = self.out_final(x)



        #pos_restricted = 1+torch.tanh(self.final_out_rel_weights)
        #final_out_sum = torch.sum(pos_restricted)




        return F.log_softmax(x, dim=1)



    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        out = self._common_step(batch, batch_idx, "train")[self.mask['train_mask']]


        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['train_mask']])
        loss = nll
        pred = out.max(1)[1]
        acc = pred.eq(y[0][self.mask['train_mask']]).sum().item() / self.mask['train_mask'].sum().item()
        #self.log("train_loss", loss)
        self.log_dict({"train_loss": loss, "train_acc":acc})

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        out = self._common_step(batch, batch_idx, "test")[self.mask['test_mask']]
        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['test_mask']])
        loss = nll
        pred = out.max(1)[1]
        acc = pred.eq(y[0][self.mask['test_mask']]).sum().item() / self.mask['test_mask'].sum().item()
        if acc > self.maxa:
            self.log_dict({"max_test_acc": acc})
            self.maxa = acc
        else:
            self.log_dict({"max_test_acc": self.maxa})
        self.log_dict({"test_loss": loss, "test_acc":acc})
        
        out = self._common_step(batch, batch_idx, "val")[self.mask['val_mask']]
        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['val_mask']])
        loss = nll
        pred = out.max(1)[1]
        acc = pred.eq(y[0][self.mask['val_mask']]).sum().item() / self.mask['val_mask'].sum().item()
        self.log_dict({"val_loss": loss, "val_acc":acc})
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
        out = self._common_step(batch, batch_idx, "test")[self.mask['test_mask']]
        _, y = batch
        nll = F.nll_loss(out, y[0][self.mask['test_mask']])
        loss = nll
        pred = out.max(1)[1]
        acc = pred.eq(y[0][self.mask['test_mask']]).sum().item() / self.mask['test_mask'].sum().item()
        self.log_dict({"test_loss": loss, "test_acc":acc})
        return loss



    def predict_step(self):
        return None


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

