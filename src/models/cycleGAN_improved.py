from models.modules import *
from models.cycleGAN import CycleGAN
from utils import helper
from evaluation.vis import *
import numpy as np
import tensorflow as tf
import pprint
import pandas as pd

#=========================================================================
#
#  This file implements our method for synthetic data generation, which 
#  consists of four CycleGANs and a classifier.       
#
#=========================================================================

class QuadCycleGAN():
    
    def __init__(self, 
                 X_source_1, X_target_1, source_name_1, target_name_1, 
                 X_source_2, X_target_2, source_name_2, target_name_2,
                 X_source_3, X_target_3, source_name_3, target_name_3,
                 X_source_4, X_target_4, source_name_4, target_name_4,
                 session_id, dim_g, dim_d, dim_c,
                 save_path, GPU_CONFIG, 
                 lr, keep_probs,batch_size, train_epochs,
                 lambda_cyc, lambda_cls, params=None):
        self.X_source_1 = X_source_1
        self.X_target_1 = X_target_1
        self.source_name_1 = source_name_1
        self.target_name_1 = target_name_1
        self.X_source_2 = X_source_2
        self.X_target_2 = X_target_2
        self.source_name_2 = source_name_2
        self.target_name_2 = target_name_2
        self.X_source_3 = X_source_3
        self.X_target_3 = X_target_3
        self.source_name_3 = source_name_3
        self.target_name_3 = target_name_3
        self.X_source_4 = X_source_4
        self.X_target_4 = X_target_4
        self.source_name_4 = source_name_4
        self.target_name_4 = target_name_4
        self.session_id = session_id
        self.dim_g = dim_g
        self.dim_d = dim_d
        self.dim_c = dim_c
        self.save_path = save_path
        self.GPU_CONFIG = GPU_CONFIG
        self.lr = lr
        self.keep_probs = keep_probs
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.lambda_cyc = lambda_cyc
        self.lambda_cls = lambda_cls
        self.params = params
        self.cgan_1 = CycleGAN(self.X_source_1, self.X_target_1, 
                               self.source_name_1, self.target_name_1, 
                               self.session_id, self.dim_g, self.dim_d, 
                               save_path=self.save_path+'/'+ \
                                           self.target_name_1+'/', 
                               GPU_CONFIG=self.GPU_CONFIG, 
                               batch_size=self.batch_size,
                               train_epochs=self.train_epochs,
                               lambda_cyc=self.lambda_cyc,
                               params=self.params[
                                   self.source_name_1+'2'+self.target_name_1
                               ])
        self.cgan_2 = CycleGAN(self.X_source_2, self.X_target_2, 
                               self.source_name_2, self.target_name_2, 
                               self.session_id, self.dim_g, self.dim_d, 
                               save_path=self.save_path+'/'+ \
                                           self.target_name_2+'/', 
                               GPU_CONFIG=self.GPU_CONFIG, 
                               batch_size=self.batch_size,
                               train_epochs=self.train_epochs,
                               lambda_cyc=self.lambda_cyc,
                               params=self.params[
                                   self.source_name_2+'2'+self.target_name_2
                               ])
        self.cgan_3 = CycleGAN(self.X_source_3, self.X_target_3, 
                               self.source_name_3, self.target_name_3, 
                               self.session_id, self.dim_g, self.dim_d, 
                               save_path=self.save_path+'/'+ \
                                           self.target_name_3+'/', 
                               GPU_CONFIG=self.GPU_CONFIG, 
                               batch_size=self.batch_size,
                               train_epochs=self.train_epochs,
                               lambda_cyc=self.lambda_cyc,
                               params=self.params[
                                   self.source_name_3+'2'+self.target_name_3
                               ])
        self.cgan_4 = CycleGAN(self.X_source_4, self.X_target_4, 
                               self.source_name_4, self.target_name_4, 
                               self.session_id, self.dim_g, self.dim_d, 
                               save_path=self.save_path+'/'+ \
                                           self.target_name_4+'/', 
                               GPU_CONFIG=self.GPU_CONFIG, 
                               batch_size=self.batch_size,
                               train_epochs=self.train_epochs, 
                               lambda_cyc=self.lambda_cyc,
                               params=self.params[
                                   self.source_name_4+'2'+self.target_name_4
                               ])
       
    
    def build(self):
        self.lr_d = tf.placeholder(tf.float32)
        self.lr_g = tf.placeholder(tf.float32)
        self.cgan_1.build(encoder_id='encoder_1', 
                          decoder_id='decoder_1', 
                          discriminator_id_A='discriminator_A_1', 
                          discriminator_id_B='discriminator_B_1')
        self.cgan_2.build(encoder_id='encoder_2', 
                          decoder_id='decoder_2', 
                          discriminator_id_A='discriminator_A_2', 
                          discriminator_id_B='discriminator_B_2')
        self.cgan_3.build(encoder_id='encoder_3', 
                          decoder_id='decoder_3', 
                          discriminator_id_A='discriminator_A_3', 
                          discriminator_id_B='discriminator_B_3')
        self.cgan_4.build(encoder_id='encoder_4', 
                          decoder_id='decoder_4', 
                          discriminator_id_A='discriminator_A_4', 
                          discriminator_id_B='discriminator_B_4')
        self.d_loss = self.cgan_1.d_loss + self.cgan_2.d_loss + \
                        self.cgan_3.d_loss + self.cgan_4.d_loss
        self.g_loss = self.cgan_1.g_loss + self.cgan_2.g_loss + \
                        self.cgan_3.g_loss + self.cgan_4.g_loss
        # define classification loss
        if self.lambda_cls > 0:
            self.keep_prob_cls = tf.placeholder(tf.float32)
            syn = tf.concat([self.cgan_1.encoding_A, 
                             self.cgan_2.encoding_A, 
                             self.cgan_3.encoding_A, 
                             self.cgan_4.encoding_A], axis=0) 
            logits = discriminator(syn, self.dim_c, 'classifier', 
                                   keep_prob=self.keep_prob_cls)
            self.labels = tf.placeholder(tf.int64, shape=(None,))
            self.c_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.labels
                )
            )
            self.g_loss += self.lambda_cls*self.c_loss
        
        
    def set_optimizer(self):
        t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(t_vars)
#         pprint.pprint(t_vars)
        d_vars = [var for var in t_vars 
                  if 'discriminator' in var.name or 'encoder' in var.name]
        g_vars = [var for var in t_vars 
                  if 'encoder' in var.name or 'decoder' in var.name or \
                  'classifier' in var.name]
        self.D_optimizer = tf.train.AdamOptimizer(
            self.lr_d).minimize(self.d_loss, var_list=d_vars)
        self.G_optimizer = tf.train.AdamOptimizer(
            self.lr_g).minimize(self.g_loss, var_list=g_vars)
        
    
    def feed_losses(self, sess, losses, epoch):
        d_l_1, g_l_1 = sess.run([self.cgan_1.d_loss, self.cgan_1.g_loss], 
                                feed_dict={
                                    self.cgan_1.input_A: self.X_source_1, 
                                    self.cgan_1.input_B: self.X_target_1, 
                                    self.cgan_1.keep_prob: 1.0
                                }
                               )
        d_l_2, g_l_2 = sess.run([self.cgan_2.d_loss, self.cgan_2.g_loss], 
                                feed_dict={
                                    self.cgan_2.input_A: self.X_source_2, 
                                    self.cgan_2.input_B: self.X_target_2, 
                                    self.cgan_2.keep_prob: 1.0
                                }
                               )
        d_l_3, g_l_3 = sess.run([self.cgan_3.d_loss, self.cgan_3.g_loss], 
                                feed_dict={
                                    self.cgan_3.input_A: self.X_source_3, 
                                    self.cgan_3.input_B: self.X_target_3, 
                                    self.cgan_3.keep_prob: 1.0
                                }
                               )
        d_l_4, g_l_4 = sess.run([self.cgan_4.d_loss, self.cgan_4.g_loss], 
                                feed_dict={
                                    self.cgan_4.input_A: self.X_source_4, 
                                    self.cgan_4.input_B: self.X_target_4, 
                                    self.cgan_4.keep_prob: 1.0
                                }
                               )
        print('CycleGAN_1 - Epoch %i: D Loss: %f, G Loss: %f;' % (epoch, 
                                                                  d_l_1, 
                                                                  g_l_1)) 
        print('CycleGAN_2 - Epoch %i: D Loss: %f, G Loss: %f;' % (epoch, 
                                                                  d_l_2, 
                                                                  g_l_2)) 
        print('CycleGAN_3 - Epoch %i: D Loss: %f, G Loss: %f;' % (epoch, 
                                                                  d_l_3, 
                                                                  g_l_3)) 
        print('CycleGAN_4 - Epoch %i: D Loss: %f, G Loss: %f;' % (epoch, 
                                                                  d_l_4, 
                                                                  g_l_4)) 
        losses['d1'].append(d_l_1)
        losses['g1'].append(g_l_1)
        losses['d2'].append(d_l_2)
        losses['g2'].append(g_l_2)
        losses['d3'].append(d_l_3)
        losses['g3'].append(g_l_3)
        losses['d4'].append(d_l_4)
        losses['g4'].append(g_l_4)
        return losses
    
    
    def generate_synthetic_samples(self, sess, save=True):
        # synthetic samples for source 1
        enc_1 = sess.run(self.cgan_1.encoding_A, 
                         feed_dict={self.cgan_1.input_A: self.X_source_1, 
                                    self.cgan_1.input_B: self.X_target_1, 
                                    self.cgan_1.keep_prob: 1.0}
                        )
        enc_1_df = pd.DataFrame(
            enc_1, index=[v+"_syn_"+self.target_name_1 
                          for v in self.X_source_1.index.values])
        enc_1_df.columns = self.X_source_1.columns.values
        if save:
            enc_1_df.to_hdf(
                self.save_path+'syn_samples/syn_'+self.session_id+'.h5', 
                key=self.target_name_1
            )
        # synthetic samples for source 2
        enc_2 = sess.run(self.cgan_2.encoding_A, 
                         feed_dict={self.cgan_2.input_A: self.X_source_2, 
                                    self.cgan_2.input_B: self.X_target_2, 
                                    self.cgan_2.keep_prob: 1.0}
                        )
        enc_2_df = pd.DataFrame(
            enc_2, index=[v+"_syn_"+self.target_name_2 
                          for v in self.X_source_2.index.values])
        enc_2_df.columns = self.X_source_2.columns.values
        if save:
            enc_2_df.to_hdf(
                self.save_path+'syn_samples/syn_'+self.session_id+'.h5', 
                key=self.target_name_2
            )
        # synthetic samples for source 3
        enc_3 = sess.run(self.cgan_3.encoding_A, 
                         feed_dict={self.cgan_3.input_A: self.X_source_3, 
                                    self.cgan_3.input_B: self.X_target_3, 
                                    self.cgan_3.keep_prob: 1.0})
        enc_3_df = pd.DataFrame(
            enc_3, index=[v+"_syn_"+self.target_name_3 
                          for v in self.X_source_3.index.values])
        enc_3_df.columns = self.X_source_3.columns.values
        if save:
            enc_3_df.to_hdf(
                self.save_path+'syn_samples/syn_'+self.session_id+'.h5', 
                key=self.target_name_3)
        # synthetic samples for source 4
        enc_4 = sess.run(self.cgan_4.encoding_A, 
                         feed_dict={self.cgan_4.input_A: self.X_source_4, 
                                    self.cgan_4.input_B: self.X_target_4, 
                                    self.cgan_4.keep_prob: 1.0}
                        )
        enc_4_df = pd.DataFrame(
            enc_4, index=[v+"_syn_"+self.target_name_4 
                          for v in self.X_source_4.index.values])
        enc_4_df.columns = self.X_source_4.columns.values
        if save:
            enc_4_df.to_hdf(
                self.save_path+'syn_samples/syn_'+self.session_id+'.h5', 
                key=self.target_name_4)
        
        
    def learn_representation(self):
        tf.reset_default_graph()        
        self.build()
        self.set_optimizer()
        t_vars = tf.trainable_variables()
        pprint.pprint(t_vars)
        # start training
        losses = {'d1': [], 'g1': [], 'd2': [], 'g2': [], 'd3': [], 'g3': [], 
                  'd4': [], 'g4': [],}
        init = tf.global_variables_initializer() 
        with tf.Session(config=self.GPU_CONFIG) as sess:
            sess.run(init)
            for epoch in range(self.train_epochs+1):
                X_source_1_shuffle = self.X_source_1.sample(frac=1)
                X_target_1_shuffle = self.X_target_1.sample(frac=1)
                X_source_2_shuffle = self.X_source_2.sample(frac=1)
                X_target_2_shuffle = self.X_target_2.sample(frac=1)
                X_source_3_shuffle = self.X_source_3.sample(frac=1)
                X_target_3_shuffle = self.X_target_3.sample(frac=1)
                X_source_4_shuffle = self.X_source_4.sample(frac=1)
                X_target_4_shuffle = self.X_target_4.sample(frac=1)
                length_source_1 = int(
                    len(X_source_1_shuffle) / self.batch_size)
                length_target_1 = int(
                    len(X_target_1_shuffle) / self.batch_size)
                length_source_2 = int(
                    len(X_source_2_shuffle) / self.batch_size)
                length_target_2 = int(
                    len(X_target_2_shuffle) / self.batch_size)
                length_source_3 = int(
                    len(X_source_3_shuffle) / self.batch_size)
                length_target_3 = int(
                    len(X_target_3_shuffle) / self.batch_size)
                length_source_4 = int(
                    len(X_source_4_shuffle) / self.batch_size)
                length_target_4 = int(
                    len(X_target_4_shuffle) / self.batch_size)
                lr_d_current = self.lr['d']*(0.8**int(epoch / 50))
                lr_g_current = self.lr['g']*(0.8**int(epoch / 50))
                if self.lambda_cls > 0:
                    labels = np.concatenate(
                        [[i]*self.batch_size for i in range(4)]
                    )
                for i in range(min([length_source_1, length_target_1, 
                                    length_source_2, length_target_2, 
                                    length_source_3, length_target_3, 
                                    length_source_4, length_target_4])):
                    X_source_1_batch = X_source_1_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_target_1_batch = X_target_1_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_source_2_batch = X_source_2_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_target_2_batch = X_target_2_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_source_3_batch = X_source_3_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_target_3_batch = X_target_3_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_source_4_batch = X_source_4_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_target_4_batch = X_target_4_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    sess.run([self.D_optimizer], 
                             feed_dict={
                                 self.cgan_1.input_A: X_source_1_batch, 
                                 self.cgan_1.input_B: X_target_1_batch,
                                 self.cgan_2.input_A: X_source_2_batch, 
                                 self.cgan_2.input_B: X_target_2_batch,
                                 self.cgan_3.input_A: X_source_3_batch, 
                                 self.cgan_3.input_B: X_target_3_batch,
                                 self.cgan_4.input_A: X_source_4_batch, 
                                 self.cgan_4.input_B: X_target_4_batch,
                                 self.cgan_1.keep_prob: self.keep_probs['d'], 
                                 self.cgan_2.keep_prob: self.keep_probs['d'],
                                 self.cgan_3.keep_prob: self.keep_probs['d'], 
                                 self.cgan_4.keep_prob: self.keep_probs['d'], 
                                 self.lr_d: lr_d_current
                             })
                    for _ in range(2):
                        feed_dict={
                            self.cgan_1.input_A: X_source_1_batch, 
                            self.cgan_1.input_B: X_target_1_batch, 
                            self.cgan_2.input_A: X_source_2_batch, 
                            self.cgan_2.input_B: X_target_2_batch,
                            self.cgan_3.input_A: X_source_3_batch, 
                            self.cgan_3.input_B: X_target_3_batch,
                            self.cgan_4.input_A: X_source_4_batch, 
                            self.cgan_4.input_B: X_target_4_batch,
                            self.cgan_1.keep_prob: self.keep_probs['g'],
                            self.cgan_2.keep_prob: self.keep_probs['g'],
                            self.cgan_3.keep_prob: self.keep_probs['g'],
                            self.cgan_4.keep_prob: self.keep_probs['g'],
                            self.lr_g: lr_g_current
                        }
                        if self.lambda_cls > 0:
                            feed_dict[self.keep_prob_cls] = self.keep_probs['c']
                            feed_dict[self.labels] = labels
                            sess.run([self.G_optimizer], feed_dict=feed_dict)
                        else:
                            sess.run([self.G_optimizer], feed_dict=feed_dict)
                                 
                if epoch % 10 == 0:
                    losses = self.feed_losses(sess, losses, epoch)
            # save losses  
            pd.DataFrame(losses).to_hdf(
                self.save_path+'losses/losses_'+self.session_id+'.h5', 
                key='quad_cycle')
            # generate synthetic samples with transferred emotions
            self.generate_synthetic_samples(sess)
            