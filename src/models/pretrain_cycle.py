import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import recall_score
import pprint
from utils import helper
from models.modules import *

#=========================================================================
#
#  This file implements pret-training for generators.       
#
#=========================================================================

class Pretrain:
    
    def __init__(self, X_source, X_target, 
                 source_name, target_name,
                 session_id, dim_g, 
                 save_path, GPU_CONFIG,
                 lr, batch_size,
                 keep_prob, train_epochs):
        self.X_source = X_source
        self.X_target = X_target
        self.source_name = source_name
        self.target_name = target_name
        self.session_id = session_id
        self.dim_g = dim_g
        self.save_path = save_path
        self.GPU_CONFIG = GPU_CONFIG
        self.lr = lr
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.train_epochs = train_epochs
        

    def train(self, encoder_id='encoder', decoder_id='decoder'):
        tf.reset_default_graph()
        self.input_A = tf.placeholder("float", 
                                      shape=[None, self.X_source.shape[1]])
        self.input_B = tf.placeholder("float", 
                                      shape=[None, self.X_target.shape[1]])
        keep_prob = tf.placeholder(tf.float32)
        # synthetic target
        self.encoding_A = encoder(self.input_A, self.dim_g, encoder_id, 
                                  keep_prob)
        # reconstructed source
        decoding_A = decoder(self.encoding_A, self.dim_g, decoder_id, 
                             keep_prob)
        # synthetic source
        encoding_B = decoder(self.input_B, self.dim_g, decoder_id, 
                             keep_prob)
        # reconstructed target
        decoding_B = encoder(encoding_B, self.dim_g, encoder_id, 
                             keep_prob)
        REC_loss_A = tf.reduce_mean(tf.pow(self.input_A - decoding_A, 2))
        REC_loss_B = tf.reduce_mean(tf.pow(self.input_B - decoding_B, 2))
        loss = REC_loss_A + REC_loss_B
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)
        # start training
        init = tf.global_variables_initializer() 
        t_vars = tf.trainable_variables()
#         pprint.pprint(t_vars)
        self.saver = tf.train.Saver()
        with tf.Session(config=self.GPU_CONFIG) as sess:
            sess.run(init)
            for epoch in range(self.train_epochs+1):
                X_source_shuffle = self.X_source.sample(frac=1)
                X_target_shuffle = self.X_target.sample(frac=1) 
                length_source = int(len(X_source_shuffle) / self.batch_size)
                length_target = int(len(X_target_shuffle) / self.batch_size)
                for i in range(min(length_source, length_target)):
                    X_source_batch = X_source_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_target_batch = X_target_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    sess.run(optimizer, 
                             feed_dict={self.input_A: X_source_batch, 
                                        self.input_B: X_target_batch, 
                                        keep_prob: self.keep_prob})
                if epoch % 100 == 0:
                    l = sess.run(loss, 
                                 feed_dict={self.input_A: self.X_source, 
                                            self.input_B: self.X_target, 
                                            keep_prob: 1.0})
                    print('Epoch %i: Train Loss: %f' % (epoch, l))
            key = self.source_name+'2'+self.target_name
            # save pre-training parameters 
            params = {}
            for var in t_vars:
                params[var.name] = var.eval() 
            np.savez_compressed(
                self.save_path+'params_'+self.session_id+'_'+key+'.npz', **params)

