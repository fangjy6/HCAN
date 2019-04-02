import numpy as np
import scipy.sparse as sp

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_adj_attribute(dataset):
    #load adj and attributes
    edge_file = open("data/{}.edge".format(dataset), 'r')
    attri_file = open("data/{}.node".format(dataset), 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_nunber:{}".format(dataset, node_num, edge_num, attribute_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))
        
    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    
    return adj,attribute
    

def load_data(dataset):
    
    adj,attribute = load_adj_attribute(dataset)
    
    if dataset in ["facebook", "DBLP"]:
        return adj, attribute, None
    
    #get label
    label = np.array(read_label(dataset)) - 1
    
    Y = np.zeros((len(label),len(np.unique(label))),dtype=np.int32)
    
    for i in range(len(label)):
        Y[i,label[i]] = 1

    print("finishing loading {}!".format(dataset))
    return adj, attribute, Y


def read_label(inputFileName):
    f = open("data/{}.label".format(inputFileName), "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    i = 0
    for line in lines:
        l = line.strip("\n\r")
        y[i] = int(l)
        i += 1
    return y