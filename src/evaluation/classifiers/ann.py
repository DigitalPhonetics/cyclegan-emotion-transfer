import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pandas as pd

#=========================================================================
#
#  Feed forward NN classifier for training, test and evaluation
#  
#  - Arguments: Training and test data
#  - Output: NN classification model, prediction, recall, confusion matrix             
#
#=========================================================================

class ANN:
    
    def __init__(self, dim_c, save_path, GPU_CONFIG, session_id, 
                 lr, batch_size, keep_prob, train_epochs, n):
        self.dim_c = dim_c
        self.save_path = save_path
        self.GPU_CONFIG = GPU_CONFIG
        self.session_id = session_id
        self.lr = lr
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.train_epochs = train_epochs
        self.n = n  # print out loss every n epochs
        print("lr: {}, batch_size: {}, keep_prob: {}, train_epochs: "
              "{}".format(lr, batch_size, keep_prob, train_epochs))
        
        
    def build(self):
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.constant_initializer(0.)
        self.inputs = tf.placeholder(tf.float32, 
                                     shape=(None, self.dim_c[0]), 
                                     name='inputs')
        self.targets = tf.placeholder(tf.int64, 
                                      shape=(None,), 
                                      name='targets')
        self.keep_prob_ = tf.placeholder(tf.float32)
        self.params = {}
        layer = self.inputs
        for i in range(1, len(self.dim_c)):
            dim_in = self.dim_c[i-1]
            dim_out = self.dim_c[i]
            weight = tf.get_variable(name='W'+str(i),
                                     shape=[dim_in, dim_out],
                                     initializer=w_init,
                                     trainable=True)
            self.params['W'+str(i)] = weight
            bias = tf.get_variable(name='b'+str(i),
                                   shape=[dim_out],
                                   initializer=b_init,
                                   trainable=True)
            self.params['b'+str(i)] = bias
            layer = tf.nn.xw_plus_b(layer, weight, bias)
            if i < len(self.dim_c)-1:
                layer = tf.nn.leaky_relu(layer)
                layer = tf.nn.dropout(layer, self.keep_prob_)
        logits = layer
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.targets
            )
        )
        self.predict_op = tf.argmax(logits, 1)
        self.saver = tf.train.Saver(self.params)
        return loss
        

    def fit(self, X_train, y_train, X_test, y_test):
        print('Size of test(dev) set: {}'.format(len(X_test)))
        tf.reset_default_graph()
        loss = self.build()
        
        train_op = tf.train.AdamOptimizer(self.lr, 
                                          beta1=0.9).minimize(loss)        
        init = tf.global_variables_initializer()   
        with tf.Session(config=self.GPU_CONFIG) as sess:
            sess.run(init)
            for epoch in range(self.train_epochs+1):
                X_shuffle = X_train.sample(frac=1)
                y_shuffle = y_train.reindex(X_shuffle.index) 
                for i in range(int(len(X_train) / self.batch_size)):
                    X_batch = X_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size
                    ]
                    y_batch = y_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size
                    ] 
                    
                    sess.run([train_op], 
                             feed_dict={self.inputs: X_batch, 
                                        self.targets: y_batch, 
                                        self.keep_prob_: self.keep_prob}) 

                if epoch % self.n == 0:
                    loss_train = sess.run(loss, 
                                          feed_dict={self.inputs: X_train, 
                                                     self.targets: y_train, 
                                                     self.keep_prob_: 1.0})
                    loss_test = sess.run(loss, 
                                         feed_dict={self.inputs: X_test, 
                                                    self.targets: y_test, 
                                                    self.keep_prob_: 1.0})
                    test_pred = sess.run(self.predict_op, 
                                         feed_dict={self.inputs: X_test, 
                                                    self.keep_prob_: 1.0})
                    test_recall = recall_score(y_test, test_pred, 
                                               average='macro')
                    print("Epoch: {} - Train Loss: {}; Test Loss: {}; "
                          "Test Recall: {}".format(epoch, loss_train, 
                                                   loss_test, test_recall))
            self.saver.save(sess, self.save_path+self.session_id+'.ckpt')  

    
    def predict(self, X_test):
        with tf.Session(config=self.GPU_CONFIG) as sess:
            self.saver.restore(sess, self.save_path+self.session_id+'.ckpt')
            pred = sess.run(self.predict_op, 
                            feed_dict={self.inputs: X_test, 
                                       self.keep_prob_: 1.0})
        return pred


    def get_recall(self, y_pred, y_test):
        recall = recall_score(y_test, y_pred, average="macro")
        return recall

    
    def get_confusion_matrix(self, y_pred, y_test):
        cm = confusion_matrix(y_test, y_pred)
        return cm




