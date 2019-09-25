from models.modules import *
from utils import helper
import numpy as np
import tensorflow as tf
import pprint
import pandas as pd

#=========================================================================
#
#  This file implements CycleGAN.       
#
#=========================================================================

class CycleGAN():
    
    def __init__(self, X_source, X_target, 
                 source_name, target_name,
                 session_id, dim_g, dim_d,
                 save_path, GPU_CONFIG, 
                 batch_size,
                 train_epochs,
                 lambda_cyc,
                 lr={'d': 2e-4, 'g': 2e-4},
                 keep_probs={'d': 0.8, 'g': 0.8},
                 params=None):
        self.X_source = X_source
        self.X_target = X_target
        self.source_name = source_name
        self.target_name = target_name
        self.session_id = session_id
        self.dim_g = dim_g
        self.dim_d = dim_d
        self.save_path = save_path
        self.GPU_CONFIG = GPU_CONFIG
        self.lr = lr
        self.keep_probs = keep_probs
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.lambda_cyc = lambda_cyc
        self.params = params
       
        
    def get_placeholder(self):
        self.input_A = tf.placeholder("float", shape=[None, self.dim_g[0]])
        self.input_B = tf.placeholder("float", shape=[None, self.dim_g[0]])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr_d = tf.placeholder(tf.float32)
        self.lr_g = tf.placeholder(tf.float32)
        
        
    def generate(self, encoder_id, decoder_id):
        self.encoding_A = encoder(self.input_A, self.dim_g, encoder_id, 
                                  keep_prob=self.keep_prob, 
                                  params=self.params)
        self.decoding_A = decoder(self.encoding_A, self.dim_g, decoder_id, 
                                  keep_prob=self.keep_prob, 
                                  params=self.params)
        self.encoding_B = decoder(self.input_B, self.dim_g, decoder_id, 
                                  keep_prob=self.keep_prob, 
                                  params=self.params)
        self.decoding_B = encoder(self.encoding_B, self.dim_g, encoder_id, 
                                  keep_prob=self.keep_prob, 
                                  params=self.params)
    
    
    def discriminate(self, discriminator_id_A, discriminator_id_B):
        self.D_fake_A = discriminator(self.encoding_A, self.dim_d, 
                                      discriminator_id_A, 
                                      keep_prob=self.keep_prob)
        self.D_real_A = discriminator(self.input_B, self.dim_d, 
                                      discriminator_id_A, 
                                      keep_prob=self.keep_prob)
        self.D_fake_B = discriminator(self.encoding_B, self.dim_d, 
                                      discriminator_id_B, 
                                      keep_prob=self.keep_prob)
        self.D_real_B = discriminator(self.input_A, self.dim_d, 
                                      discriminator_id_B, 
                                      keep_prob=self.keep_prob)
    
    
    def compute_loss(self):
        REC_loss_A = tf.reduce_mean(
            tf.pow(self.input_A - self.decoding_A, 2)
        )
        D_loss_fake_A = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_A, 
                labels=tf.zeros_like(self.D_fake_A)
            )
        )
        D_loss_real_A = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_real_A,
                labels=tf.ones_like(self.D_real_A)*0.9
            )
        )
        D_loss_A = (D_loss_real_A + D_loss_fake_A) / 2.
        G_loss_A = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_A, 
                labels=tf.ones_like(self.D_fake_A)*0.9
            )
        )
        # define stat_loss (mean_and_std_loss) to further narrow 
        # the distance between target and synthetic dataset
#         mean_syn, var_syn = tf.nn.moments(self.encoding_A, axes=[0])
#         mean_real, var_real = tf.nn.moments(self.input_B, axes=[0])
#         self.stat_loss = tf.reduce_mean(
#             tf.pow(mean_syn - mean_real, 2)) + tf.reduce_mean(
#             tf.pow(var_syn - var_real, 2)
#         )
       
        REC_loss_B = tf.reduce_mean(
            tf.pow(self.input_B - self.decoding_B, 2)
        )
        D_loss_fake_B = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_B, 
                labels=tf.zeros_like(self.D_fake_B)
            )
        )
        D_loss_real_B = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_real_B,
                labels=tf.ones_like(self.D_real_B)*0.9
            )
        )
        D_loss_B = (D_loss_real_B + D_loss_fake_B) / 2.
        G_loss_B = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_B, 
                labels=tf.ones_like(self.D_fake_B)*0.9
            )
        )
        self.REC_loss = REC_loss_A + REC_loss_B
        self.D_loss = D_loss_A + D_loss_B
        self.G_loss = G_loss_A + G_loss_B
        self.d_loss = self.D_loss
        self.g_loss = self.G_loss + self.lambda_cyc*self.REC_loss 
#        + self.stat_loss
        
    
    def set_optimizer(self, encoder_id='encoder', decoder_id='decoder', 
                      discriminator_id_A='discriminator_A', 
                      discriminator_id_B='discriminator_B'):
        t_vars = tf.trainable_variables()
        # update discriminator and encoder to minimize d loss 
        d_vars = [var for var in t_vars 
                  if discriminator_id_A in var.name or \
                  discriminator_id_B in var.name or \
                  encoder_id in var.name]
        # update encoder and decoder to minimize g loss
        g_vars = [var for var in t_vars 
                  if encoder_id in var.name or \
                  decoder_id in var.name]
        self.D_optimizer = tf.train.AdamOptimizer(
            self.lr_d).minimize(self.d_loss, var_list=d_vars)
        self.G_optimizer = tf.train.AdamOptimizer(
            self.lr_g).minimize(self.g_loss, var_list=g_vars)
    
    
    def feed_losses(self, sess, losses, epoch):
        d_l, g_l = sess.run([self.d_loss, self.g_loss], 
                            feed_dict={self.input_A: self.X_source, 
                                       self.input_B: self.X_target, 
                                       self.keep_prob: 1.0})
        if epoch % 10 == 0:
            print('Epoch %i: D Loss: %f, G Loss: %f;' % (epoch, d_l, g_l)) 
        losses['d'].append(d_l)
        losses['g'].append(g_l)
        return losses
    
    
    def generate_synthetic_samples(self, sess, identifier="A"):
        if identifier == "A":
            x = self.X_source
            key = self.source_name + '2' + self.target_name
            encoding = self.encoding_A
        else:
            x = self.X_target
            key = self.target_name + '2' + self.source_name
            encoding = self.encoding_B
        enc = sess.run(encoding, feed_dict={self.input_A: self.X_source, 
                                            self.input_B: self.X_target, 
                                            self.keep_prob: 1.0})
        enc_df = pd.DataFrame(
            enc, index=[v+"_syn_"+key for v in x.index.values])
        enc_df.columns = x.columns.values
        enc_df.to_hdf(
            self.save_path+'syn_samples/syn_'+self.session_id+'.h5', 
            key=key)
                
    
    def build(self, encoder_id='encoder', decoder_id='decoder', 
              discriminator_id_A='discriminator_A', 
              discriminator_id_B='discriminator_B'):
        self.get_placeholder()
        self.generate(encoder_id, decoder_id)
        self.discriminate(discriminator_id_A, discriminator_id_B)
        self.compute_loss()
    
    
    def learn_representation(self):
        tf.reset_default_graph()
        self.build()        
        self.set_optimizer()  
        t_vars = tf.trainable_variables()
#         pprint.pprint(t_vars)
        self.saver = tf.train.Saver(t_vars)
        # start training
        losses = {'d': [], 'g': []}
        init = tf.global_variables_initializer() 
        with tf.Session(config=self.GPU_CONFIG) as sess:
            sess.run(init)
            for epoch in range(self.train_epochs+1):
                X_source_shuffle = self.X_source.sample(frac=1)
                X_target_shuffle = self.X_target.sample(frac=1)
                length_source = int(len(X_source_shuffle) / self.batch_size)
                length_target = int(len(X_target_shuffle) / self.batch_size)
                # decaying learning rate
                lr_d_current = self.lr['d']*(0.8**int(epoch / 50))
                lr_g_current = self.lr['g']*(0.8**int(epoch / 50))
                for i in range(min(length_source, length_target)):
                    X_source_batch = X_source_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    X_target_batch = X_target_shuffle.iloc[
                        i*self.batch_size : (i+1)*self.batch_size]
                    sess.run([self.D_optimizer], 
                             feed_dict={self.input_A: X_source_batch, 
                                        self.input_B: X_target_batch, 
                                        self.keep_prob: self.keep_probs['d'], 
                                        self.lr_d: lr_d_current})
                    for _ in range(2):
                        sess.run(
                            [self.G_optimizer], 
                            feed_dict={self.input_A: X_source_batch, 
                                       self.input_B: X_target_batch, 
                                       self.keep_prob: self.keep_probs['g'], 
                                       self.lr_g: lr_g_current}
                        )
                losses = self.feed_losses(sess, losses, epoch)
            # save losses  
            key = self.source_name+'2'+self.target_name
#             pd.DataFrame(losses).to_hdf(
#                 self.save_path+'losses/losses_'+self.session_id+'.h5', key=key)
            # generate synthetic samples with transferred emotions
            self.generate_synthetic_samples(sess, "A") # syn of target
#             self.generate_synthetic_samples(sess, "B") # syn of source
            