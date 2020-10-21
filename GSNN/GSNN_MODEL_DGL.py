import torch
from torch import nn
from torch.nn import functional as F

from torch.distributions import Normal
import scipy.sparse as sp
import numpy as np

from dgl.nn.pytorch import GraphConv
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GSNN_Decoder_GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual
                 ):
        super(GSNN_Decoder_GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.x_to_h = nn.Linear(in_dim, num_hidden)
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            num_hidden*2, num_hidden*2, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1]*2, num_hidden*2, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden*2 * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        self.dropout = nn.Dropout(p=0.9)

    def forward(self, inputs, z = None):
        inputs = self.dropout(inputs)
        h = self.x_to_h(inputs)
        h = F.relu(h)
        z = z.unsqueeze(0).repeat(inputs.shape[0], 1)
        h = torch.cat((h, z), dim = 1)
        h = h / (torch.norm(h, dim=1).view(-1,1)+0.000001)

        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits


class GSNN_Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, z_dim, model_descr='GNP'):
        super(GSNN_Decoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.model_descr = model_descr

        self.x_to_h1 = nn.Linear(x_dim, h_dim)
        self.h1_to_h = nn.Linear(h_dim+z_dim, h_dim+z_dim)
        self.h1_to_h2 = nn.Linear(h_dim+z_dim, h_dim+z_dim)
        self.h_to_y = nn.Linear(h_dim+z_dim, y_dim)
        self.dropout = nn.Dropout(p=0.9)
        
    def forward(self, x, adj, z = None):
        x = self.dropout(x)
        x = x / (torch.sum(x,dim=1) + 0.0001).view(-1, 1)
        h = self.x_to_h1(x)
        h = F.relu(h)
        
        z = z.unsqueeze(0).repeat(x.shape[0], 1)
        h = torch.cat((h, z), dim = 1)
        h = h / (torch.norm(h, dim=1).view(-1,1)+0.000001)

        h = self.h1_to_h2(torch.spmm(adj, h))
        h = F.relu(h)
        y = self.h_to_y(torch.spmm(adj, h))
        
        return y

class GSNN_Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, r_dim):
        super(GSNN_Encoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim

        self.xy_to_h = nn.Linear(h_dim + y_dim, r_dim)
        self.h_to_r = nn.Linear(h_dim, r_dim)

    def forward(self, x, y):
        xy = torch.cat((x, y), dim = 1)
        h = self.xy_to_h(xy)
        h = F.relu(h)
        r = self.h_to_r(h)
        return r

class R_to_MuSigma(nn.Module):
    def __init__(self, r_dim, z_dim):
        super(R_to_MuSigma, self).__init__()
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.r_to_h = nn.Linear(r_dim, r_dim)
        self.h_to_mu = nn.Linear(r_dim, z_dim)
        self.h_to_sigma = nn.Linear(r_dim, z_dim)

    def forward(self, r):
        h = self.r_to_h(r)
        h = F.relu(h)
        mu = self.h_to_mu(h)
        sigma = 0.1 + 0.9 * F.sigmoid(self.h_to_sigma(h))
        return mu, sigma
        
class GSNN(nn.Module):
    def __init__(self, x_dim, y_dim, h_dim, r_dim, z_dim, x_to_yu, x_to_y):
        super(GSNN, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.x_to_yu = x_to_yu
        self.xy_to_r = GSNN_Encoder(x_dim, y_dim, h_dim, r_dim)
        self.r_to_musigma = R_to_MuSigma(r_dim, z_dim)
        # self.x_to_y = GSNN_Decoder(x_dim, h_dim, y_dim, z_dim)
        self.x_to_y = x_to_y

        if torch.cuda.is_available():
            self.x_to_yu = self.x_to_yu.cuda()
            self.xy_to_r = self.xy_to_r.cuda()
            self.r_to_musigma = self.r_to_musigma.cuda()
            self.x_to_y = self.x_to_y.cuda()
    
    def get_z_dim(self):
        return self.z_dim

    def r_aggregate(self, r):
        return torch.mean(r, dim = 0)

    def forward(self, x, y_, adj, non_label):
        if self.training:
            x = x.contiguous().view(-1, self.x_dim)
            y_ = y_.contiguous().view(-1, self.y_dim)
            # x2y
            y_encode, y_embedding = self.x_to_yu(x)

            q_dis_total, y_total, y_pred_total_total = [], [], []
            for i in range(1):
                y = y_.clone()
                y[non_label] = F.gumbel_softmax(y_encode[non_label], tau=1.0, hard=True)
                y_total.append(y)
                # encode
                r_nodes = self.xy_to_r(y_embedding, y)
                r_graph = self.r_aggregate(r_nodes)
                mu, sigma = self.r_to_musigma(r_graph)

                q_dis = Normal(mu, sigma)
                q_dis_total.append(q_dis)
                y_pred_total = []
                for _ in range(1):
                    z_sample = q_dis.rsample()
                    #Decode
                    y_pred = self.x_to_y(x, z_sample)
                    y_pred_total.append(y_pred)
                y_pred_total_total.append(y_pred_total)
                
            return y_pred_total_total, q_dis_total, y_total, y_encode
        else:
            x = x.contiguous().view(-1, self.x_dim)
            y_ = y_.contiguous().view(-1, self.y_dim)
            y_encode, y_embedding = self.x_to_yu(x)
            y_pred_total = []
            for i in range(40):
                y = y_.clone()
                y[non_label] = F.gumbel_softmax(y_encode[non_label], tau=1.0, hard=True)

                # encode
                r_nodes = self.xy_to_r(y_embedding, y)
                r_graph = self.r_aggregate(r_nodes)
                mu, sigma = self.r_to_musigma(r_graph)

                q_dis = Normal(mu, sigma)
                for _ in range(1):
                    z_sample = q_dis.rsample()
                    #Decode
                    y_pred = self.x_to_y(x, z_sample)
                    y_pred_total.append(y_pred)

            y_pred = sum(y_pred_total)/len(y_pred_total)
            
            return y_pred, y_encode
        
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 as_encoder=False):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.as_encoder=as_encoder

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers-1):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        h = self.dropout(h)
        logits = layer(self.g, h)
        if self.as_encoder:
            return logits, h
        else:
            return logits

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 as_encoder=False,
                 normalize=False):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        self.as_encoder=as_encoder
        self.normalize=normalize

    def forward(self, inputs):
        h = inputs
        h = h / (torch.sum(h,dim=1) + 0.0001).view(-1, 1)
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        h = h.view(h.shape[0],50,-1).mean(dim=2)
        if self.as_encoder:
            return logits, h
        else:
            return logits