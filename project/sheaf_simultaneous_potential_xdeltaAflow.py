from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch_sparse
import torch.nn.functional as F
from torch_sparse import transpose, spmm, spspmm
# Quick and dirty solution, refactor later





class sheaf_gradient_flow_potential_xdeltaA(pl.LightningModule):
    """docstring for sheaf_diffusion"""
    def __init__(self, graph, Nv, dv, Ne, de, layers, input_dim, output_dim, channels, left_weights, right_weights, potential, mask, use_act, augmented, add_lp, add_hp, dropout, input_dropout, perturb_diagonal, free_potential, stalk_mixing, channel_mixing, learning_rate = 0.01, weight_decay = 5e-4):
        super(sheaf_gradient_flow_potential_xdeltaA, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.perturb_diagonal = perturb_diagonal
        self.free_potential = free_potential
        self.stalk_mixing = stalk_mixing
        self.channel_mixing = channel_mixing
        self.graph  = graph
        self.layers = layers
        self.hidden_channels = channels

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

        # Copy weights from sheaf(coboundary) learner
        #self.register_buffer("sheaf_laplacian", sheaf_laplacian)
    
        #MAKE NN BACKBONE MORE INTERESTING MLP
        self.coboundary_vec_out = nn.ParameterList([nn.Parameter(torch.randn(de, dv)) for i in range(self.Ne)])

        # Coboundary maps for vertex where edge is coming in
        self.coboundary_vec_in  = nn.ParameterList([nn.Parameter(torch.randn(de, dv)) for i in range(self.Ne)])
        for i in range(self.Ne):
            nn.init.orthogonal_(self.coboundary_vec_in[i])
            nn.init.orthogonal_(self.coboundary_vec_out[i])

        # Linear Matrics Stack more Layers
        # APPEND ADJACENCY MATRIX TO IT 
        self.graph_to_sheaf = nn.Linear(input_dim, self.final_d*channels)
        self.graph_to_sheaf2  = nn.Linear(self.final_d*channels, self.final_d*channels)
        self.graph_to_sheaf3  = nn.Linear(self.final_d*channels, self.final_d*channels)
        self.self_energy_channel_mixing = nn.Linear(channels, channels)
        self.self_energy_stalk_mixing   = nn.Linear(self.final_d, self.final_d)
        self.diffusion_stalk_mixing     = nn.Linear(self.final_d, self.final_d)
        self.diffusion_channel_mixing   = nn.Linear(channels, channels)


        self.mask = mask
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.use_act = use_act

        # ADD NEW OPTIONS
        self.augmented = augmented
        self.add_lp    = add_lp
        self.add_hp    = add_hp 
        


        self.lin_left_weights  = nn.ModuleList()
        self.lin_right_weights = nn.ModuleList()

        self.potential = nn.Parameter(torch.randn(self.Ne), requires_grad=True)
        
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

        for i in range(self.Ne):
            nn.init.normal_(self.coboundary_vec_out[i])
            nn.init.normal_(self.coboundary_vec_in[i])
        #self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        #if self.second_linear:
        #    self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)


    def build_laplacian_from_vector(self):

        coboundary = torch.zeros((self.Ne*self.de, self.Nv*self.dv))

        for i, (v1, v2) in enumerate(self.graph.edges):
            coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]

        sheaf_laplacian = torch.matmul(coboundary.t(), coboundary)

        return sheaf_laplacian


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

    def normalize_coboundary(self, coboundary, add_diagonal):
        laplacian = torch.matmul(coboundary.t(), coboundary)
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

        normalized_coboundary =  coboundary @ D_pow
        return normalized_coboundary


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
        #print("dpow",D_pow.size())
        #print("lap", laplacian.size())
        normalized_laplacian = D_pow * laplacian * D_pow




        return normalized_laplacian


    def _common_step(self, batch, batch_idx, stage: str):

        #print(batch)
        if stage == 'train':
            training = True
        else:
            training = False
        batch, _  = batch
        batch = F.dropout(batch, p=self.input_dropout, training=training)
        
        batch = self.graph_to_sheaf(batch)
        batch = F.elu(batch)
        batch = F.dropout(batch, p=self.dropout, training=training)
        batch = self.graph_to_sheaf2(batch)
        batch = F.elu(batch)
        batch = F.dropout(batch, p=self.dropout, training=training)
        batch = self.graph_to_sheaf3(batch)
        #print(batch.size())
        #print(self.hidden_dim, self.graph_size, self.final_d)
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
        coboundary = torch.zeros((self.Ne*self.de, self.Nv*self.dv))

        for i, (v1, v2) in enumerate(self.graph.edges):
            coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]

        #coboundary = coboundary.to_sparse()
        coboundary = coboundary.to(batch)
        #print(self.potential)
        if self.free_potential == False:
            #potential_blocks = [torch.diag(1+self.potential[i].to(batch)*torch.ones(self.de).to(batch)) for i in range(self.Ne)]
            potential_blocks = [torch.diag((torch.FloatTensor(1).uniform_(1, 1.01).to(batch)+torch.tanh(self.potential[i])).to(batch)*torch.ones(self.de).to(batch)) for i in range(self.Ne)]
            #eps = torch.FloatTensor(self.dv * self.Nv).uniform_(1, 1.01).to(batch)
            #potential_blocks = [torch.diag((self.potential[i]*self.potential[i]).to(batch)*torch.ones(self.de).to(batch)) for i in range(self.Ne)]
            #potential_blocks = [torch.diag((torch.sigmoid(self.potential[i])).to(batch)*torch.ones(self.de).to(batch)) for i in range(self.Ne)]
        if self.free_potential == True:
            potential_blocks = [torch.diag(self.potential[i].to(batch)*torch.ones(self.de).to(batch)) for i in range(self.Ne)]
        potential_matrix = torch.block_diag(*potential_blocks)
        
        potential_times_coboundary = torch.matmul(potential_matrix, coboundary).to(batch)
        sheaf_laplacian = torch.matmul(coboundary.t(), potential_times_coboundary).to(batch)
        if stage == 'train' and self.perturb_diagonal == True:
            eps = torch.FloatTensor(self.dv * self.Nv).uniform_(-0.001, 0.001).to(batch)
            eps.requires_grad = False
            lap_add_diagonal = torch.diag(1.+eps)
                #lap_add_diagonal = torch.eye(self.dv * self.Nv).to(x_diffusion)

        else:
            lap_add_diagonal = torch.eye(self.dv * self.Nv).to(batch)

        sheaf_laplacian = self.normalize_laplacian(sheaf_laplacian, lap_add_diagonal)
        for layer in range(self.layers):
            #if layer == 0 or self.nonlinear:
            #    x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
            #    maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
            #    L, trans_maps = self.laplacian_builder(maps)
            #    self.sheaf_learners[layer].set_L(trans_maps)


            x = F.dropout(x, p=self.dropout, training=training)


            if self.left_weights and self.right_weights:
                x_diffusion = self.left_right_linear(x, self.lin_left_weights[layer], self.lin_right_weights[layer])
            elif not self.left_weights and self.right_weights:
                x_diffusion=self.lin_right_weights[layer](x)
            else:
                x_diffusion = x

            # Use the adjacency matrix rather than the diagonal
            #x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            #print(x.size())
            #print(self.sheaf_laplacian.size())

            
            #cob_indices = torch.zeros((2,2*self.Ne*self.de*self.dv), dtype=torch.long)
            #cob_values  = torch.zeros((2*self.Ne*self.de*self.dv,), dtype=torch.float)
            #count = 0
            #for i, (v1, v2) in enumerate(self.graph.edges):
            #    for j, e in enumerate(range(self.de*i,self.de*(i+1))):
            #        for k, v in enumerate(range(self.dv*v1,self.dv*(v1+1))):
            #            cob_indices[0, count]  = e
            #            cob_indices[1, count]  = v
            #            cob_values[count] = 1 * self.coboundary_vec_out[i][j, k]
            #            count += 1
            #        for l, v in enumerate(range(self.dv*v2,self.dv*(v2+1))):
            #            cob_indices[0, count]  = e
            #            cob_indices[1, count]  = v
            #            cob_values[count] = -1 * self.coboundary_vec_in[i][j, l]
            #            count += 1

            #coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            #coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]


            #cob_indices = cob_indices.to(x_diffusion).type(torch.long)
            #cob_values = cob_values.to(x_diffusion)
            #cob_T_indices, cob_T_values = transpose(cob_indices, cob_values, self.Ne*self.de, self.Nv*self.dv)



            #sheaf_laplacian_indices, sheaf_laplacian_values = spspmm(cob_T_indices, cob_T_values, cob_indices, cob_values, self.Nv*self.dv, self.Ne*self.de, self.Nv*self.dv)
            #sheaf_laplacian_values = sheaf_laplacian_values.to(x_diffusion)

            #perturbation for the values

            
            
            

            #sheaf_laplacian = torch.sparse.mm(coboundary.t(),coboundary).to(x_diffusion)



            

            
            
            #potential_times_coboundary = torch.matmul(potential_matrix, coboundary).to(x_diffusion)
            #sheaf_laplacian = torch.matmul(coboundary.t(), potential_times_coboundary).to(x_diffusion)

            if self.add_lp or self.add_hp:
                if self.add_lp and self.add_hp:
                    new_sheaf_laplacian = torch.zeros((self.Nv*(self.dv+2), self.Nv*(self.dv+2)))

                    for i in range(self.Nv):
                        new_sheaf_laplacian[i*(self.dv+2):i*(self.dv+2)+self.dv , i*(self.dv+2):i*(self.dv+2)+self.dv ] = sheaf_laplacian[i*self.dv:(i+1)*self.dv,i*self.dv:(i+1)*self.dv]
                        new_sheaf_laplacian[i*(self.dv+2)+self.dv , i*(self.dv+2)+self.dv ] = 1.
                        new_sheaf_laplacian[i*(self.dv+2)+self.dv+1 , i*(self.dv+2)+self.dv+1 ] = -1.


                    for i, (v1, v2) in enumerate(self.graph.edges):
                        new_sheaf_laplacian[v1*(self.dv+2):v1*(self.dv+2)+self.dv , v2*(self.dv+2):v2*(self.dv+2)+self.dv ] = sheaf_laplacian[v1*self.dv:(v1+1)*self.dv,v2*self.dv:(v2+1)*self.dv]
                        new_sheaf_laplacian[v1*(self.dv+2)+self.dv , v2*(self.dv+2)+self.dv ] = 1.
                        new_sheaf_laplacian[v1*(self.dv+2)+self.dv+1 , v2*(self.dv+2)+self.dv+1 ] = -1.

                        new_sheaf_laplacian[v2*(self.dv+2):v2*(self.dv+2)+self.dv , v1*(self.dv+2):v1*(self.dv+2)+self.dv ] = sheaf_laplacian[v2*self.dv:(v2+1)*self.dv,v1*self.dv:(v1+1)*self.dv]
                        new_sheaf_laplacian[v2*(self.dv+2)+self.dv, v1*(self.dv+2)+self.dv ] = 1.
                        new_sheaf_laplacian[v2*(self.dv+2)+self.dv+1 , v1*(self.dv+2)+self.dv+1 ] = -1.




                elif self.add_lp:
                    new_sheaf_laplacian = torch.zeros((self.Nv*(self.dv+1), self.Nv*(self.dv+1)))

                    for i in range(self.Nv):
                        new_sheaf_laplacian[i*(self.dv+1):i*(self.dv+1)+self.dv , i*(self.dv+1):i*(self.dv+1)+self.dv ] = sheaf_laplacian[i*self.dv:(i+1)*self.dv,i*self.dv:(i+1)*self.dv]
                        new_sheaf_laplacian[i*(self.dv+1)+self.dv , i*(self.dv+1)+self.dv ] = -1.


                    for i, (v1, v2) in enumerate(self.graph.edges):
                        new_sheaf_laplacian[v1*(self.dv+1):v1*(self.dv+1)+self.dv , v2*(self.dv+1):v2*(self.dv+1)+self.dv ] = sheaf_laplacian[v1*self.dv:(v1+1)*self.dv,v2*self.dv:(v2+1)*self.dv]
                        new_sheaf_laplacian[v1*(self.dv+1)+self.dv , v2*(self.dv+1)+self.dv ] = -1.

                        new_sheaf_laplacian[v2*(self.dv+1):v2*(self.dv+1)+self.dv , v1*(self.dv+1):v1*(self.dv+1)+self.dv ] = sheaf_laplacian[v2*self.dv:(v2+1)*self.dv,v1*self.dv:(v1+1)*self.dv]
                        new_sheaf_laplacian[v2*(self.dv+1)+self.dv, v1*(self.dv+1)+self.dv ] = -1.

                elif self.add_hp:
                    new_sheaf_laplacian = torch.zeros((self.Nv*(self.dv+1), self.Nv*(self.dv+1)))

                    for i in range(self.Nv):
                        new_sheaf_laplacian[i*(self.dv+1):i*(self.dv+1)+self.dv , i*(self.dv+1):i*(self.dv+1)+self.dv ] = sheaf_laplacian[i*self.dv:(i+1)*self.dv,i*self.dv:(i+1)*self.dv]
                        new_sheaf_laplacian[i*(self.dv+1)+self.dv , i*(self.dv+1)+self.dv ] = 1.


                    for i, (v1, v2) in enumerate(self.graph.edges):
                        new_sheaf_laplacian[v1*(self.dv+1):v1*(self.dv+1)+self.dv , v2*(self.dv+1):v2*(self.dv+1)+self.dv ] = sheaf_laplacian[v1*self.dv:(v1+1)*self.dv,v2*self.dv:(v2+1)*self.dv]
                        new_sheaf_laplacian[v1*(self.dv+1)+self.dv , v2*(self.dv+1)+self.dv ] = 1.

                        new_sheaf_laplacian[v2*(self.dv+1):v2*(self.dv+1)+self.dv , v1*(self.dv+1):v1*(self.dv+1)+self.dv ] = sheaf_laplacian[v2*self.dv:(v2+1)*self.dv,v1*self.dv:(v1+1)*self.dv]
                        new_sheaf_laplacian[v2*(self.dv+1)+self.dv, v1*(self.dv+1)+self.dv ] = 1.



                sheaf_laplacian = new_sheaf_laplacian.to(x_diffusion)

            x_mixed = x_diffusion
            x_diffusion = torch.matmul(sheaf_laplacian, x_diffusion)



            #x_diffusion = spmm(sheaf_laplacian_indices, sheaf_laplacian_values, self.Nv*self.dv,self.Nv*self.dv, x_diffusion)



            
            #coboundary = torch.zeros((self.Ne*self.de, self.Nv*self.dv))

            #for i, (v1, v2) in enumerate(self.graph.edges):
            #    coboundary[self.de*i:self.de*(i+1),self.dv*v1:self.dv*(v1+1)] = 1. * self.coboundary_vec_out[i]
            #    coboundary[self.de*i:self.de*(i+1),self.dv*v2:self.dv*(v2+1)] = -1.* self.coboundary_vec_in[i]

            #coboundary = coboundary.to_sparse()
            #sheaf_laplacian = torch.matmul(coboundary.t(), coboundary).to(x_diffusion)
            #sheaf_laplacian = torch.sparse.mm(coboundary.t(),coboundary).to(x_diffusion)

            #x_diffusion = torch.matmul(sheaf_laplacian, x_diffusion)
            #x_diffusion = torch.sparse.mm(sheaf_laplacian, x_diffusion).to_dense()
            x_update = x_diffusion
            if self.stalk_mixing == True:
                x_stalk_mixing   =  self.self_energy_stalk_mixing(x.reshape(-1, self.final_d)).reshape(-1, self.hidden_channels)           
                x_update += x_stalk_mixing
            if self.channel_mixing == True:
                x_channel_mixing = self.self_energy_channel_mixing(x.reshape(-1, self.hidden_channels))
                x_update += x_channel_mixing
                


            if self.use_act:
                x_update = F.elu(x_update)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x_update
            lap_update = torch.matmul(x_mixed, x.t())
            #print('sheaflaplacian', sheaf_laplacian.size())
            #print('lapupdate', lap_update.size())
            
            if self.use_act:
                lap_update = F.elu(lap_update)
            sheaf_laplacian_new = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) *sheaf_laplacian - 0.5*lap_update
            x = x0
            sheaf_laplacian = sheaf_laplacian_new

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
        pred = out.max(1)[1]
        acc = pred.eq(y[0][self.mask['train_mask']]).sum().item() / self.mask['train_mask'].sum().item()
        #self.log("train_loss", loss)
        self.log_dict({"train_loss": loss, "train_acc":acc})

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forwardset
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

