import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score,average_precision_score

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def self_loops(adj):
    
    diagonal = adj.diagonal()
    non_zero = diagonal.nonzero()[0]
    # remove sparse value in the diagonal
    for i in non_zero:
        adj[i,i] = 0
		
    adj = adj + sp.eye(adj.shape[0])
        
    return adj

    
def preprocess_graph(adj):
    """preprocess graph for GCN:"""
    if len(adj.diagonal().nonzero()[0]) == adj.shape[0]:
        adj_ = adj
    else:
        adj_ = self_loops(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def saveEmbed(dataset, node_z_mean, node_z_var, fea_z_mean, fea_z_var):
    node_z_mean = node_z_mean.numpy()
    node_z_var = node_z_var.numpy()
    fea_z_mean = fea_z_mean.numpy()
    fea_z_var = fea_z_var.numpy()
    np.save("result/{}.node.z.mean".format(dataset), node_z_mean)
    np.save("result/{}.node.z.var".format(dataset), node_z_var)
    np.save("result/{}.fea.z.mean".format(dataset), fea_z_mean)
    np.save("result/{}.fea.z.var".format(dataset), fea_z_var)
    print("successfully saved!")
	
	

def standardize_data(f, train_mask):
    """Standardize feature matrix """
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return features.todense(), sparse_to_tuple(features)
    return features

def sigmoid(x):
    return 1. / (1+np.exp(-1*x))

def get_roc_score(reconstruction, edge_pos, edge_neg, shape, logits=True):
    
    reconstruction = sigmoid(reconstruction) if logits else reconstruction
    reconstruction = reconstruction.reshape(shape)
    preds_pos = []
    for edge in edge_pos:
        preds_pos.append(reconstruction[edge[0],edge[1]])
    
    preds_neg = []
    for edge in edge_neg:
        preds_neg.append(reconstruction[edge[0],edge[1]])
    
    preds_all = np.hstack([preds_pos, preds_neg])
    true_all = np.hstack([np.ones(len(preds_pos)),np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(true_all,preds_all)
    ap_score = average_precision_score(true_all,preds_all)
    
    return roc_score,ap_score


def mask_test_edges(adj, p_val=0.05, p_test=0.10):
    """
    adj is an adjacant matrix (scipy sparse matrix)
    
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
    adj_train : training adjacant matrix
    train_edges : array indicating the training edges
    val_edges : array indicating the validation edges
    val_edge_false: array indicating the false edges in validation dataset
    """
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    
    #get deges from adjacant matrix
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]])
        edges_dic[(adj_row[i], adj_col[i])] = 1
    
    #split the dataset into training,validation and test dataset
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) * p_test))
    num_val = int(np.floor(len(edges) * p_val))
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    
    
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_edges_false) < num_test :
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val :
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val :
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test :
                    test_edges_false.append([i, j])
    
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false



def mask_test_feas(features,p_val=0.05, p_test=0.10):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
    false_feas_dic = {}
    #num_test = int(np.floor(len(feas) / 10.))
    #num_val = int(np.floor(len(feas) / 20.))
    num_val = round(len(feas)*p_val)
    num_test = round(len(feas)*p_test)
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test or len(val_feas_false) < num_val:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333 :
            if len(test_feas_false) < num_test :
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val :
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val :
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test :
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])), shape=features.shape)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false

