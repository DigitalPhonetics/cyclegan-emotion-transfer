from models.modules import *
from utils import helper
import numpy as np
import tensorflow as tf
import pprint
import pandas as pd

#=========================================================================
#
#  This file re-implements the adversarial autoencoder in the paper of 
#  Sahu et al. 2017             
#
#=========================================================================

class AAE:
    
    def __init__(self, X_train, y_train, X_test, y_test, 
                 session_id, dim_g, dim_d,
                 save_path,
                 GPU_CONFIG,
                 label_incorporated=True,
                 learning_rate={'rec': 1e-4, 'd': 1e-4, 'g': 1e-4}, 
                 dropout={'rec': 0.5, 'd': 0.8, 'g': 0.8}, 
                 batch_size=64, 
                 train_epochs=500):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.session_id = session_id
        self.dim_g = dim_g
        self.dim_d = dim_d
        self.save_path = save_path
        self.GPU_CONFIG = GPU_CONFIG
        self.label_incorporated = label_incorporated
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.train_epochs = train_epochs
    
    
    def get_placeholder(self):
        X = tf.placeholder("float", shape=[None, self.dim_g[0]])
        Z = tf.placeholder('float', shape=[None, self.dim_g[-1]])
        keep_prob = tf.placeholder(tf.float32)
        if self.label_incorporated:
            Labels = tf.placeholder('float', shape=[None, 4])
        else:
            Labels = None
        return X, Z, keep_prob, Labels
    
    
    def generate(self, X, encoder_id, decoder_id, keep_prob):
        encoding = encoder(X, self.dim_g, encoder_id, keep_prob)
        decoding = decoder(encoding, self.dim_g, decoder_id, keep_prob)
        return encoding, decoding
    
    
    def discriminate(self, encoding, Z, discriminator_id, keep_prob, Labels):
        if Labels is not None:
            D_fake = discriminator(tf.concat([encoding, Labels], axis=1), 
                                   self.dim_d, discriminator_id, 
                                   keep_prob, C=4)
            D_real = discriminator(tf.concat([Z, Labels], axis=1), 
                                   self.dim_d, discriminator_id, 
                                   keep_prob, C=4)
        else:
            D_fake = discriminator(encoding, self.dim_d, discriminator_id, 
                                   keep_prob)
            D_real = discriminator(Z, self.dim_d, discriminator_id, 
                                   keep_prob)
        return D_fake, D_real
    
    
    def get_loss(self, X, decoding, D_fake, D_real):
        REC_loss = tf.reduce_mean(tf.pow(X - decoding, 2))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=D_fake, 
                                  labels=tf.zeros_like(D_fake))
                               )
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                  logits=D_real,
                                  labels=tf.ones_like(D_real))
                               )
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=D_fake, 
                                labels=tf.ones_like(D_fake))
                               )
        return REC_loss, D_loss, G_loss
    
    
    def get_optimizer(self, REC_loss, D_loss, G_loss):
        t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(t_vars)
#         pprint.pprint(t_vars)
        rec_vars = [var for var in t_vars 
                    if 'encoder' in var.name or 'decoder' in var.name]
        d_vars = [var for var in t_vars 
                  if 'discriminator' in var.name or 'encoder' in var.name]
        g_vars = [var for var in t_vars if 'encoder' in var.name]  
        REC_optimizer = tf.train.AdamOptimizer(
            self.learning_rate['rec']).minimize(REC_loss, var_list=rec_vars)
        D_optimizer = tf.train.AdamOptimizer(
            self.learning_rate['d']).minimize(D_loss, var_list=d_vars)
        G_optimizer = tf.train.AdamOptimizer(
            self.learning_rate['g']).minimize(G_loss, var_list=g_vars)
        return REC_optimizer, D_optimizer, G_optimizer
  

    def feed_code_vector(self, sess, encoding, X, keep_prob, for_test=False):
        if for_test:
            x = self.X_test
            y = self.y_test
            name = 'test'
        else:
            x = self.X_train
            y = self.y_train
            name = 'train'
        enc = sess.run(encoding, feed_dict={X: x, keep_prob: 1.0})
        enc_df = pd.DataFrame(enc, index=x.index.values)
        enc_df.to_csv(self.save_path+'code_vectors/'+name+'/'+ \
                      self.session_id+'.csv', sep='\t')
    
    
    def feed_losses(self, sess, REC_loss, D_loss, G_loss, losses, X, Z, 
                    keep_prob, epoch, Labels, for_test=False):
        if for_test:
            x = self.X_test
            y = self.y_test
        else:
            x = self.X_train
            y = self.y_train
        feed_dict = {X: x, 
                     Z: helper.gaussian_mixture(y), 
                     keep_prob: 1.0}
        if Labels is not None:
            feed_dict[Labels] = helper.onehot(y)
        rec_l, d_l, g_l = sess.run([REC_loss, D_loss, G_loss], 
                                   feed_dict=feed_dict)
        if epoch % 10 == 0 and for_test==False:
            print("Epoch %i: Reconstruction Loss: %f, Discrimnator Loss: "
                  "%f, Generator Loss: %f" % (epoch, rec_l, d_l, g_l)) 
        losses['rec'].append(rec_l)
        losses['d'].append(d_l)
        losses['g'].append(g_l)
        return losses
    
    
    def learn_representation(self):
        tf.reset_default_graph()
        X, Z, keep_prob, Labels = self.get_placeholder()
        encoding, decoding = self.generate(X, 'encoder', 'decoder', 
                                           keep_prob)
        D_fake, D_real = self.discriminate(encoding, Z, 'discriminator', 
                                           keep_prob, Labels) 
        REC_loss, D_loss, G_loss = self.get_loss(X, decoding, D_fake, D_real)
        REC_optimizer, D_optimizer, G_optimizer = self.get_optimizer(
            REC_loss, D_loss, G_loss)       
        # start training
        train_losses = {'rec': [], 'd': [], 'g': []}
        test_losses = {'rec': [], 'd': [], 'g': []}
        init = tf.global_variables_initializer() 
        with tf.Session(config=self.GPU_CONFIG) as sess:
            sess.run(init)
            for epoch in range(self.train_epochs+1):
                X_shuffle = self.X_train.sample(frac=1)
                y_shuffle = self.y_train.reindex(X_shuffle.index)  
                for i in range(int(len(self.X_train) / self.batch_size)):
                    X_batch = X_shuffle.iloc[i*self.batch_size : 
                                             (i+1)*self.batch_size]
                    y_batch = y_shuffle.iloc[i*self.batch_size : 
                                             (i+1)*self.batch_size] 
                    z_samples = helper.gaussian_mixture(y_batch)
                    
                    feed_dict_rec = {X: X_batch, 
                                     keep_prob: self.dropout['rec']}
                    feed_dict_d = {X: X_batch, Z: z_samples, 
                                   keep_prob: self.dropout['d']}
                    feed_dict_g = {X: X_batch, keep_prob: self.dropout['g']}
                    
                    if self.label_incorporated:
                        labels = helper.onehot(y_batch)
                        feed_dict_d[Labels] = labels
                        feed_dict_g[Labels] = labels
                        
                    sess.run(REC_optimizer, feed_dict=feed_dict_rec)
#                     for _ in range(2):
                    sess.run(D_optimizer, feed_dict=feed_dict_d)
                    for _ in range(10):
                        sess.run(G_optimizer, feed_dict=feed_dict_g)
                            
                train_losses = self.feed_losses(
                    sess, REC_loss, D_loss, G_loss, train_losses, 
                    X, Z, keep_prob, epoch, Labels, for_test=False)
                test_losses = self.feed_losses(
                    sess, REC_loss, D_loss, G_loss, test_losses, 
                    X, Z, keep_prob, epoch, Labels, for_test=True)
            # save training and test losses 
            pd.DataFrame(train_losses).to_csv(
                self.save_path+'losses/train/losses_'+self.session_id+'.csv', 
                sep='\t')
            pd.DataFrame(test_losses).to_csv(
                self.save_path+'losses/test/losses_'+self.session_id+'.csv', 
                sep='\t')
            # feed encoding with training data                                                                      
            self.feed_code_vector(sess, encoding, X, keep_prob)  
             # feed encoding with test data
            self.feed_code_vector(sess, encoding, X, keep_prob, 
                                  for_test=True)
            # save model parameters
            self.saver.save(
                sess, self.save_path+'gan_params/'+self.session_id+'.ckpt')
        
    
    def generate_synthetic_samples(self, code_vectors):
        # select 4*100 code vectors randomly
        samples = []
        labels = []
        for j in range(4):
            s = code_vectors[self.y_train==j].sample(100)
            samples.append(s)
            labels.append(self.y_train.loc[s.index.values]) 
        X_samples = pd.concat(samples)
        # restore model parameters
        with tf.Session(config=self.GPU_CONFIG) as sess:
            self.saver.restore(
                sess, self.save_path+'gan_params/'+self.session_id+'.ckpt')
            t_vars = tf.trainable_variables()
#             pprint.pprint(t_vars)
            # extract decoder parameters
            dec_vars = [var for var in t_vars if 'decoder' in var.name]
            dec_params = {}
            for var in dec_vars:
                dec_params[var.name] = var.eval()            
            # pass through the decoder part
            layer = tf.convert_to_tensor(X_samples.values, 
                                         dtype=np.float32)
            for i in range(1, len(self.dim_g)):
                weight = tf.convert_to_tensor(
                    dec_params['decoder/W'+str(i)+':0'])
                bias = tf.convert_to_tensor(
                    dec_params['decoder/b'+str(i)+':0']
                )
                layer = tf.nn.xw_plus_b(layer, weight, bias)
                if i < len(self.dim_g)-1:
                    layer = tf.nn.leaky_relu(layer)           
            X_syn = pd.DataFrame(
                layer.eval(), 
                index=[v+"_syn" for v in X_samples.index.values]
            ) 
            X_syn.columns = self.X_train.columns.values
            # save synthetic samples
            X_syn.to_csv(
                self.save_path+'syn_samples/syn_'+self.session_id+'.csv', 
                sep='\t')