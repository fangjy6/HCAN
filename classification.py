import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
import torch
from hyperspherical_vae.distributions import VonMisesFisher


def get_classification_score(y_pred, y_true):
    
    macro_f1 = f1_score(y_pred, y_true, average="macro")
    micro_f1 = f1_score(y_pred, y_true, average="micro")
    accuracy = accuracy_score(y_pred, y_true, normalize=True)
    
    return macro_f1, micro_f1, accuracy

    
def classify(dataset, labels):
    
    """classification using node embeddings and logistic regression"""
    print('classification using lr :dataset:', dataset)
    
    #load node embeddings
    y = np.argmax(labels, axis=1)
    node_z_mean = np.load("result/{}.node.z.mean.npy".format(dataset))
    node_z_var = np.load("result/{}.node.z.var.npy".format(dataset))
    
    q_z = VonMisesFisher(torch.tensor(node_z_mean), torch.tensor(node_z_var))

    #train the model and get metrics
    macro_f1_avg = 0
    micro_f1_avg = 0
    acc_avg = 0
        
    
    for i in range(10):
        #sample data
        node_embedding = q_z.rsample()
        node_embedding = node_embedding.numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(node_embedding, y, train_size=0.2, test_size=1000, random_state=2019)

        clf = LogisticRegression(solver="lbfgs", multi_class="multinomial",random_state=0).fit(X_train,y_train)
        
        y_pred = clf.predict(X_test)
    
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        micro_f1 = f1_score(y_test, y_pred, average="micro")
        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1
        acc_avg += accuracy
    
    return macro_f1_avg/10, micro_f1_avg/10, acc_avg/10
