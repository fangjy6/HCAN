
import numpy as np
import torch
import argparse
from model4 import ModelVAE, VAEloss
from input_data import load_data
from utils import preprocess_graph,preprocess_features,saveEmbed
from utils import train_val_test_split_adjacency,mask_test_feas,mask_test_edges,get_roc_score,self_loops
from logger import Logger

#parameters setting
parser = argparse.ArgumentParser(description="s_co_emb settings")
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--distribution', type=str, default='normal')#'vmf' or 'normal'
parser.add_argument('--h_dim', type=int, default=64, help='number of units in hidden layer')
parser.add_argument('--z_dim', type=int, default=32, help='dimensions of latent variables')
parser.add_argument('--alpha', type=int, default=1, help='parameter of classification')
parser.add_argument('--beta', type=int, default=1, help='parameter of features')
parser.add_argument('--epochs', type=int, default=200, help='training epochs')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
parser.add_argument('--temp', type=float, default=1, help='parameter of gumbel softmax')
args = parser.parse_args()
       
#data preprocessing
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(args.dataset)
#adj = self_loops(adj)
#make train,val and test edge and features
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = train_val_test_split_adjacency(adj,undirected=True,p_val=0.05, p_test=0.10,connected=False)
#adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, pval=0.05, ptest=0.10)
fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features,p_val=0.05,p_test=0.10)

adj_orig = torch.Tensor(adj_train.todense().astype(np.float32))
features_orig = torch.Tensor(fea_train.todense().astype(np.float32))
adj = preprocess_graph(adj_train) # prepare the graph for GCN
adj = torch.Tensor(adj.todense())
features = preprocess_features(fea_train)
features = torch.Tensor(features.todense())

num_nodes = adj.shape[0]
num_feas = features.shape[1]
num_labels = y_train.shape[1]
pos_weight_u = (torch.tensor(1. * num_nodes * num_nodes) - adj_orig.sum()) / adj_orig.sum()
pos_weight_a = (torch.tensor(1. * num_nodes * num_feas) - features_orig.sum()) /features_orig.sum()


model = ModelVAE(num_labels, num_feas, num_nodes, args.h_dim, args.z_dim, activation=torch.relu,
                 distribution=args.distribution, temperature=args.temp, dropout=args.dropout)
loss_op = VAEloss(args.alpha, args.beta)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


#train the model
logit = True if args.distribution == "normal" else False
logger = Logger('./logs/{}'.format(args.dataset))

for i in range(args.epochs):
    #compute loss:
    optimizer.zero_grad()
    output = model(features, adj, train_mask, y_train)
    loss, recon, kl = loss_op(model, adj_orig, features_orig, train_mask, y_train, pos_weight_u, pos_weight_a)
    #loss = loss_op(model, adj_orig, features_orig, train_mask, y_train, pos_weight_u, pos_weight_a)
    #print(loss, recon, kl)
    loss.backward()
    optimizer.step()
    
    edge_recon = output.detach().numpy()
    #print(edge_recon.shape, attr_recon.shape)
    edge_roc,edge_ap = get_roc_score(edge_recon, val_edges, val_edges_false, shape=(num_nodes,num_nodes), logits=logit)
    
    print("epoch {}: loss:{:.4f}, recon:{:.4f}, kl:{:.4f}".format(i,loss,recon,kl), end=", ")
    #print("epoch {}: loss:{:.4f}".format(i,loss), end=", ")
    print("val_edge_roc: {:.4f}, val_edge_ap:{:.4f}".format(edge_roc, edge_ap))
    
    # Tensorboard Logging
    info = {'loss':loss.item(), 'recon':recon.item(), 'kl':kl.item()}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, i+1)
    #log values and gradients of the parameters(histogram)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), i+1)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i+1)
    logger.histo_summary("recon", torch.sigmoid(output).detach().numpy(), i+1)


with torch.no_grad():
    
    edge_recon= model(features, adj, train_mask, y_train)
    edge_roc,edge_ap = get_roc_score(edge_recon, test_edges, test_edges_false, shape=(num_nodes,num_nodes), logits=logit)
    
    print('Test edge ROC score:{:.4f}'.format(edge_roc))
    print('Test edge AP score:{:.4f}'.format(edge_ap))
    

"""
import matplotlib.pyplot as plt

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1 + np.exp(-x))
    
edge_recon = model.edge_recon.detach().numpy()
edge_recon = sigmoid(edge_recon)
plt.hist(edge_recon, bins=20)
edge_recon = edge_recon.reshape([num_nodes, num_nodes])

edge_pos_pred = []
edge_neg_pred = []

for e in val_edges:
    edge_pos_pred.append(edge_recon[e[0], e[1]])
plt.figure()
plt.hist(edge_pos_pred)

for e in val_edges_false:
    edge_neg_pred.append(edge_recon[e[0], e[1]])
plt.hist(edge_neg_pred)


"""
    
    
    
    