from __future__ import division
from __future__ import print_function

import time,os
import numpy as np
import tensorflow as tf
#import scipy.sparse as sp
import matplotlib.pyplot as plt


from model import ModelVAE,OptimizerVAE
from input_data import load_data
from utils import preprocess_graph,preprocess_features
from classification import classify
#from classification import classify
#from plotscatter import plot

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

#settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('distribution', 'vmf', 'distribution string')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_multi_integer("encoder",[128, 32],"layers of encoders")#the last one is the dimension of latent variables
flags.DEFINE_multi_integer("decoder",[64, 128, 256],"layers of decoders")#the last one is the dimension of latent variables

flags.DEFINE_float("alpha",100,"parameter of the label prediction loss")
flags.DEFINE_float("beta",1e-3,"parameter of the mutual information of y and z")
flags.DEFINE_float("temperature",1,"temperature of gumbel softmax")
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')

flags.DEFINE_integer('features',1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')

#data preprocessing
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data(FLAGS.dataset)
adj_orig = adj.todense()
features_orig = features.todense()
adj = preprocess_graph(adj) # prepare the graph for GCN
adj = adj.todense()
features = preprocess_features(features)
features = features.todense()

num_nodes = adj.shape[0]
num_feas = features.shape[1]
num_labels = y_train.shape[1]
features_nonzero = len(features.nonzero()[0])

#define placeholders
placeholders = {
    'adj': tf.placeholder(tf.float32,shape=(num_nodes,num_nodes)),
    'features': tf.placeholder(tf.float32,shape = (None,num_feas)),
    'adj_orig': tf.placeholder(tf.float32, shape=(None, num_nodes)),
    'features_orig':tf.placeholder(tf.float32, shape=(None,num_feas)),
    'label':tf.placeholder(tf.float32,shape=(None,num_labels))
}

model = ModelVAE(placeholders, train_mask, num_feas, num_nodes, num_labels, activation=tf.nn.relu, distribution=FLAGS.distribution)
optimizer = OptimizerVAE(model)
#model = HGI(placeholders,node_features,num_nodes,num_feas,features_nonzero,pos_weight_u,pos_weight_a)

#train the model
#optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
#opt_op = optimizer.minimize(model.loss)
sess = tf.Session()

#grad_sum_op = tf.summary.merge([tf.summary.histogram(g[1].name,g[0]) for g in optimizer.gradients])
#summary_op = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter('log/cora_summary',sess.graph)

sess.run(tf.global_variables_initializer())

#feed_dict = {placeholders['features']:features,placeholders['adj']:adj}
feed_dict = {placeholders['features']:features, placeholders['adj']:adj,
             placeholders['adj_orig']:adj_orig, placeholders['features_orig']:features_orig, 
             placeholders['label']:y_train}

ng_losses = []
fn_losses = []
losses = []
vlss_min = np.inf
curr_step = 0
outputs = []
for epoch in range(FLAGS.epochs):
    
    t = time.time()
    
    #_,output = sess.run([optimizer.train_step,optimizer.print], feed_dict=feed_dict)
    #summary,gradient,_,output,z,y,y_z_logits = sess.run([summary_op, optimizer.gradients, optimizer.train_step,optimizer.print, model.z, model.y, model.y_z_logits], feed_dict=feed_dict)
   # _,output = sess.run([optimizer.train_step, optimizer.print], feed_dict=feed_dict)
    sess.run(optimizer.train_step, feed_dict=feed_dict)
    #print(epoch, sess.run({**optimizer.print}, feed_dict=feed_dict))
    output = sess.run({**optimizer.print}, feed_dict=feed_dict)
    #train_writer.add_summary(summary, epoch)
    
    losses.append(output['loss'])
    
    print(epoch,':',output)

    '''
    if epoch % 10 == 0:
        print(output)
       
    
    if losses[-1] < vlss_min:
        vlss_min = losses[-1]
        curr_step = 0
    else:
        curr_step += 1
        if curr_step == FLAGS.early_stopping:
            print("Early Stopping...")
            break
    '''    
    if epoch > FLAGS.early_stopping and losses[-1] > np.mean(losses[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break
    
        
print("Optimization Finished!")
#train_writer.close()


#plot the training process
'''
losses = [outputs[i]['loss'] for i in range(len(outputs))]
ELBO = [outputs[i]['ELBO'] for i in range(len(outputs))]
recon_loss = [outputs[i]['recon_loss'] for i in range(len(outputs))]
kl = [outputs[i]['KL'] for i in range(len(outputs))]
closs = [outputs[i]['closs'] for i in range(len(outputs))]
q_y_z = [outputs[i]['q_y_z'] for i in range(len(outputs))]
entropy = [outputs[i]['entropy'] for i in range(len(outputs))]


fig = plt.figure()
ax = fig.add_subplot(111)
x_axis = np.arange(len(outputs))
ax.plot(x_axis,losses,label="loss")
ax.plot(x_axis,ELBO,label="ELBO")
ax.plot(x_axis,recon_loss,label="recon_loss")
ax.plot(x_axis,kl,label="KL")
ax.plot(x_axis,closs,label="class_loss")
ax.plot(x_axis,q_y_z,label="q_y_z")
ax.plot(x_axis,entropy,label="entropy")
ax.legend()
fig.show()
'''

#save the embeddings
#z_mean, z_var, y_pred_logits = sess.run([model.z_mean,model.z_var, model.y_pred_logits], feed_dict=feed_dict)
#np.save("result/{}.z_mean".format(FLAGS.dataset, z_mean))
#np.save("result/{}.z_var".format(FLAGS.dataset, z_var))
#np.save("result/{}.y_pred_logits".format(FLAGS.dataset, y_pred_logits))
z, y_pred_logits, node_embed = sess.run([model.z, model.y_pred_logits, model.node_embed], feed_dict=feed_dict)
np.save("result/{}.node.embedding".format(FLAGS.dataset), np.concatenate((node_embed,z), axis=1))


#classification using discriminator
y_exp = np.exp(y_pred_logits)[train_mask,:]
y_pred = np.argmax(y_exp,axis = 1)
y_true = np.argmax(labels[train_mask,:], axis=1)
print("train accuracy:{}".format(np.mean(y_pred==y_true)))

y_exp = np.exp(y_pred_logits)[val_mask,:]
y_pred = np.argmax(y_exp,axis = 1)
y_true = np.argmax(labels[val_mask,:], axis=1)
print("valdation accuracy:{}".format(np.mean(y_pred==y_true)))


#classification using node embeddings and SVM)
macro_f1_avg,micro_f1_avg,accuracy = classify(FLAGS.dataset,y_train, y_val, train_mask, val_mask)
#macro_f1_avg,micro_f1_avg,accuracy = classify(FLAGS.dataset,y_train, y_test, train_mask, test_mask)
print("classification using node embeddings")
print("macro_f1: {:.4f}\nmiciro_f1: {:.4f}\naccuracy:{:.4f}".format(macro_f1_avg,micro_f1_avg,accuracy))

"""
#plot embedding
#y_true = labels.argmax(axis = 1)
#plot(FLAGS.dataset,y_true)
"""