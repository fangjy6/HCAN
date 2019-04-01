
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform


class ModelVAE(torch.nn.Module):
    
    def __init__(self, num_feas, num_nodes, h_dim, z_dim, activation=F.relu, dropout=0.5):
        """
        ModelVAE initializer
        :param num_feas: number of features in the network
        :param num_nodes: number of nodes in the network
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        """
        super(ModelVAE, self).__init__()
        
        self.num_feas = num_feas
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.activation = activation
        self.dropout = dropout
        
        #encoder
        self.node_encoder = node_encoder(self.num_feas, self.h_dim, self.z_dim,
                                         self.activation, self.dropout)
        
        self.fea_encoder = fea_encoder(self.num_nodes, self.h_dim, self.z_dim,
                                       self.activation, self.dropout)
    
        self.decoder = decoder()
        
    
    def forward(self, x, a): 
        """
        network forward calculation
        :param x: node attribute matrix
        :param a: adjacency matrix of graph
        """
    
        # encode
        self.node_z_mean, self.node_z_var = self.node_encoder(x, a)
        self.fea_z_mean, self.fea_z_var = self.fea_encoder(x)
        
        #reparameterize
        self.node_q_z, self.node_p_z = self.reparameterize(self.node_z_mean, self.node_z_var)
        self.fea_q_z, self.fea_p_z = self.reparameterize(self.fea_z_mean, self.fea_z_var)
        
        #sample z from approximate posterior
        node_z = self.node_q_z.rsample()
        fea_z = self.fea_q_z.rsample()
        
        self.edge_recon, self.fea_recon = self.decoder(node_z, fea_z)

        return self.node_z_mean, self.node_z_var, self.fea_z_mean, self.fea_z_var, self.edge_recon, self.fea_recon
    
    
    def reparameterize(self, z_mean, z_var):
        
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(self.z_dim - 1)
        
        return q_z, p_z
    

class node_encoder(torch.nn.Module):
    
    def __init__(self, in_dim, h_dim, z_dim, activation=F.relu, dropout=0.5):
        
        super(node_encoder, self).__init__()
        
        self.activation = activation
        
        #hidden layers
        bias = False
        self.fc_e0 = nn.Linear(in_dim, h_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.fc_e0.weight)
        self.drop0 = nn.Dropout(p=dropout)
        
    
        # compute mean and concentration of the von Mises-Fisher
        self.fc_mean = nn.Linear(h_dim, z_dim, bias=bias)
        torch.nn.init.xavier_uniform_(self.fc_mean.weight)
        self.fc_var = nn.Linear(h_dim, 1, bias=bias)
        torch.nn.init.xavier_uniform_(self.fc_var.weight)

        
    def forward(self, x, a):
        
        h = self.drop0(self.activation(self.fc_e0(torch.matmul(a, x))))
        
        #get latent variable z
        z_mean = self.fc_mean(torch.matmul(a, h))
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.fc_var(torch.matmul(a, h))) + 1

            
        return z_mean, z_var
    
    
    
class fea_encoder(torch.nn.Module):
    
    def __init__(self, in_dim, h_dim, z_dim, activation=F.relu, dropout=0.5):
        
        super(fea_encoder, self).__init__()
        
        self.activation = activation
        
        #hidden layers
        self.fc_e0 = nn.Linear(in_dim, h_dim)
        torch.nn.init.xavier_uniform_(self.fc_e0.weight)
        self.drop0 = nn.Dropout(p=dropout)
        
        # compute mean and concentration of the von Mises-Fisher
        self.fc_mean = nn.Linear(h_dim, z_dim)
        torch.nn.init.xavier_uniform_(self.fc_mean.weight)
        self.fc_var = nn.Linear(h_dim, 1)
        torch.nn.init.xavier_uniform_(self.fc_var.weight)
            
        
    def forward(self, x):
        
        x = torch.transpose(x, 0, 1)
        
        #hidden layers
        h = self.drop0(self.activation(self.fc_e0(x)))
        
        #get latent variable z
        z_mean = self.fc_mean(h)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.fc_var(h)) + 1
        
        return z_mean, z_var


class decoder(torch.nn.Module):
    
    def __init__(self):
        super(decoder, self).__init__()
        
    def forward(self, node_z, fea_z):
        
        node_recon = torch.matmul(node_z, torch.transpose(node_z, 0, 1))
        fea_recon = torch.matmul(node_z, torch.transpose(fea_z, 0, 1))
        
        return node_recon.view([-1]), fea_recon.view([-1])
    

class VAEloss(torch.nn.Module):
    
    def __init__(self, alpha):
        
        super(VAEloss, self).__init__()
        
        self.alpha = alpha
    
    def forward(self, model, adj_orig, fea_orig, edge_pos_weight, fea_pos_weight):
    
        num_nodes = adj_orig.shape[0]
        num_feas = adj_orig.shape[1]
        #reconstruction loss
        adj_orig = adj_orig.view([-1])
        fea_orig = fea_orig.view([-1])
        adj_recon_loss = nn.BCEWithLogitsLoss(pos_weight=edge_pos_weight)(model.edge_recon, adj_orig)
        fea_recon_loss = nn.BCEWithLogitsLoss(pos_weight=fea_pos_weight)(model.fea_recon, fea_orig)
        #print("loss: distribution:", model.distribution)
        recon_loss = self.alpha * adj_recon_loss + (1 -self.alpha) * fea_recon_loss
        
        # kl divergence
        adj_kl = (0.5 / num_nodes) * torch.distributions.kl.kl_divergence(model.node_q_z, model.node_p_z).mean()
        fea_kl = (0.5 / num_feas) * torch.distributions.kl.kl_divergence(model.fea_q_z, model.fea_p_z).mean()
             
        kl = self.alpha * adj_kl + (1 - self.alpha) * fea_kl
        
        
        loss = recon_loss + kl
       
        return loss, recon_loss, kl
    