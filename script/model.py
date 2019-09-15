import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from rnn import dynamic_att_rnn
from utils import *
from Dice import dice

class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling = False):
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.lr = tf.placeholder(tf.float64, [])
            self.use_negsampling =use_negsampling
        
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        
        if self.use_negsampling:
            self.neg_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='neg_mid_batch_ph')
            self.neg_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='neg_cat_batch_ph')
            self.neg_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.neg_mid_batch_ph)
            self.neg_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.neg_cat_batch_ph)
            self.neg_item_his_eb = tf.concat(
                [self.neg_mid_his_batch_embedded[:, :, 0, :], self.neg_cat_his_batch_embedded[:, :, 0, :]], -1)# 0 means only using the first negative item ID. 3 item IDs are inputed in the original input.
            self.neg_item_his_eb = tf.reshape(self.neg_item_his_eb,
                                                [-1, tf.shape(self.neg_mid_his_batch_embedded)[1], 32])# cat embedding 18 concate item embedding 18.
            self.neg_his_eb = tf.concat([self.neg_mid_his_batch_embedded, self.neg_cat_his_batch_embedded], -1)
            self.neg_his_eb_sum_1 = tf.reduce_sum(self.neg_his_eb, 2)
            self.neg_his_eb_sum = tf.reduce_sum(self.neg_his_eb_sum_1, 1)

    def build_fcn_net(self, inp, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, input, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=input, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat


    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
                self.neg_mid_batch_ph: inps[9],
                self.neg_cat_batch_ph: inps[10],
            })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.lr: inps[8],
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7],
                self.neg_mid_batch_ph: inps[8],
                self.neg_cat_batch_ph: inps[9],
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.target_ph: inps[6],
                self.seq_len_ph: inps[7]
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                                          ATTENTION_SIZE,
                                                          use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp, use_dice=False)

class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE,
                                        use_negsampling)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum,
                         self.item_eb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp, use_dice=False)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                           ATTENTION_SIZE,
                                           use_negsampling)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DIEN, self).__init__(n_uid, n_mid, n_cat,
                                                          EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                          use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.neg_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True, scope="att")
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

class Model_DIEN_with_InnerAtt(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
     super(Model_DIEN_with_InnerAtt, self).__init__(n_uid, n_mid, n_cat,
                                                                       EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                                           use_negsampling)
            
     # RNN layer(-s)
     with tf.name_scope('rnn_1'):
         rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                      sequence_length=self.seq_len_ph, dtype=tf.float32,
                                      scope="gru1")
         tf.summary.histogram('GRU_outputs', rnn_outputs)
                                         
     aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                      self.neg_item_his_eb[:, 1:, :],
                                      self.mask[:, 1:], stag="loss")
     self.aux_loss = aux_loss_1
                                                 
     # Attention layers
     with tf.name_scope('Attention_layer_1'):
         att_outputs1, alphas1 = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                   softmax_stag=1, stag='1_1', mode='LIST',
                                                       return_alphas=True, scope="att1")
         tf.summary.histogram('alpha1_outputs', alphas1)
                                                                 
     self.item_embedding = dim_expand(self.item_eb, rnn_outputs, scope="itemexp")
                                                                
     with tf.name_scope('rnn_3'):
          new_input2 = tf.concat([rnn_outputs, self.item_embedding], axis=-1)
          rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=new_input2,
                 att_scores = tf.expand_dims(alphas1, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru3")
          tf.summary.histogram('GRU3_Final_State', final_state2)
                                                                                 
     inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1)
         
     self.build_fcn_net(inp, use_dice=True)

class Model_DHIN_without_InnerAtt(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DHIN_without_InnerAtt, self).__init__(n_uid, n_mid, n_cat,
                                                                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                                              use_negsampling)
            
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
                                          
        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.neg_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="loss")
        self.aux_loss = aux_loss_1
                                                  
        # Attention layers
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                     softmax_stag=1, stag='1_1', mode='LIST',
                                                        return_alphas=True, scope="att1")
            tf.summary.histogram('alpha1_outputs', alphas1)
                                                          
        with tf.name_scope('Attention_layer_2'):
            att_outputs2, alphas2 = din_fcn_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                                     softmax_stag=1, stag='2_1', mode='LIST',
                                                         return_alphas=True, scope="att2")
            tf.summary.histogram('alpha2_outputs', alphas2)
        
        with tf.name_scope('rnn_2'):
            rnn_outputs1, final_state1 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs = self.item_his_eb, att_scores = tf.expand_dims(alphas2, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state1)
                                                                          
        with tf.name_scope('rnn_3'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs = rnn_outputs, att_scores = tf.expand_dims(alphas1, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru3")
            tf.summary.histogram('GRU3_Final_State', final_state2)
                                                                                  
        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state1, final_state2], 1)
            
        self.build_fcn_net(inp, use_dice=True)


class Model_DHIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DHIN, self).__init__(n_uid, n_mid, n_cat,
                                                              EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE,
                                                              use_negsampling)
        
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)
    
        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.item_his_eb[:, 1:, :],
                                         self.neg_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="loss")
        self.aux_loss = aux_loss_1

        # Attention layers
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = din_fcn_attention(self.item_eb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                     softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True, scope="att1")
            tf.summary.histogram('alpha1_outputs', alphas1)
            
        with tf.name_scope('Attention_layer_2'):
            att_outputs2, alphas2 = din_fcn_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask,
                                                     softmax_stag=1, stag='2_1', mode='LIST', return_alphas=True, scope="att2")
            tf.summary.histogram('alpha2_outputs', alphas2)

        self.item_embedding = dim_expand(self.item_eb, rnn_outputs, scope="itemexp")
        with tf.name_scope('rnn_2'):
            new_input1 = tf.concat([self.item_his_eb, self.item_embedding], axis=-1)
            rnn_outputs1, final_state1 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=new_input1, att_scores = tf.expand_dims(alphas2, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state1)
        
        with tf.name_scope('rnn_3'):
            new_input2 = tf.concat([rnn_outputs, self.item_embedding], axis=-1)
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=new_input2, att_scores = tf.expand_dims(alphas1, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru3")
            tf.summary.histogram('GRU3_Final_State', final_state2)

        inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state1, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)

