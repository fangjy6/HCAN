
import numpy as np
import torch
import argparse
from model import ModelVAE,VAEloss
from input_data import load_data
from utils import preprocess_graph,preprocess_features,saveEmbed
from utils import mask_test_feas, mask_test_edges, get_roc_score
from classification import classify


#parameters setting
parser = argparse.ArgumentParser(description="s_co_embedding settings")
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--activation', type=str, default='relu')#'rle' or 'tanh‘
parser.add_argument('--h_dim', type=int, default=512, help='number of units in hidden layer')
parser.add_argument('--z_dim', type=int, default=20, help='dimensions of latent variables')
parser.add_argument('--alpha', type=float, default=0.8, help='parameter balancing the reconstruction accuracy between edge and attribute')
parser.add_argument('--epochs', type=int, default=250, help='training epochs')
parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
args = parser.parse_args()
       
#data preprocessing
adj, features, labels = load_data(args.dataset)
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, p_val=0.05, p_test=0.10)
fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features, p_val=0.05, p_test=0.10)

adj_orig = torch.Tensor(adj_train.todense().astype(np.float32))
features_orig = torch.Tensor(fea_train.todense().astype(np.float32))
adj = preprocess_graph(adj_train) # prepare the graph for GCN
adj = torch.Tensor(adj.todense().astype(np.float32))
features = preprocess_features(fea_train)# Normalizing features of nodes of graph
features = torch.Tensor(features.todense().astype(np.float32))

num_nodes = adj.shape[0]
num_feas = features.shape[1]
if labels is not None:
    num_labels = labels.shape[1]
pos_weight_u = (torch.tensor(1. * num_nodes * num_nodes) - adj_orig.sum()) / adj_orig.sum()
pos_weight_a = (torch.tensor(1. * num_nodes * num_feas) - features_orig.sum()) / features_orig.sum()

if args.activation == "relu":
    act = torch.relu
elif args.activation == "tanh":
    act = torch.tanh
else:
    raise NotImplementedError

model = ModelVAE(num_feas, num_nodes, args.h_dim, args.z_dim, activation=act, dropout=args.dropout)
loss_op = VAEloss(args.alpha)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


#train the model
for i in range(args.epochs):
    #compute loss:
    optimizer.zero_grad()
    output = model(features, adj)
    loss, recon, kl = loss_op(model, adj_orig, features_orig, pos_weight_u, pos_weight_a)
    
    loss.backward()
    optimizer.step()
    
    edge_recon = output[-2].view([-1]).detach().numpy()
    attr_recon = output[-1].view([-1]).detach().numpy()
    #print(edge_recon.shape, attr_recon.shape)
    edge_roc,edge_ap = get_roc_score(edge_recon,val_edges, val_edges_false,shape=(num_nodes,num_nodes), logits=True)
    attr_roc,attr_ap = get_roc_score(attr_recon,val_feas, val_feas_false,shape=(num_nodes,num_feas), logits=True)
    
    print("epoch {}: loss:{:.4f}, recon:{:.4f}, kl:{:.4f}".format(i,loss,recon,kl), end=", ")
    print("val_edge_roc: {:.4f}, val_edge_ap:{:.4f}, val_attr_roc:{:.4f}, val_attr_ap:{:.4f}".format(edge_roc, edge_ap, attr_roc, attr_ap))
    

#evaluate the model
with torch.no_grad():
    
    node_z_mean, node_z_var, fea_z_mean, fea_z_var, edge_recon, attr_recon = model(features, adj)
    
    edge_roc,edge_ap = get_roc_score(edge_recon, test_edges, test_edges_false, shape=(num_nodes,num_nodes), logits=True)
    attr_roc,attr_ap = get_roc_score(attr_recon, test_feas, test_feas_false,shape=(num_nodes,num_feas), logits=True)
    
    print('Test edge ROC score:{:.4f}'.format(edge_roc))
    print('Test edge AP score:{:.4f}'.format(edge_ap))
    print('Test attr ROC score:{:.4f}'.format(attr_roc))
    print('Test attr AP score:{:.4f}'.format(attr_ap))
    
    
    saveEmbed(args.dataset, node_z_mean, node_z_var, fea_z_mean, fea_z_var)
    
    #classification using node embedding
    if args.dataset not in ["facebook", "DBLP"]:
        macro_f1_avg, micro_f1_avg, accuracy = classify(args.dataset, labels)
        print("macro_f1: {:.4f}\nmicro_f1: {:.4f}\naccuracy: {:.4f}".format(macro_f1_avg,micro_f1_avg,accuracy))
    
    
    
    
    