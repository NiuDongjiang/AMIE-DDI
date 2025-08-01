import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, AveragePooling1D,AveragePooling2D
from tensorflow.keras.initializers import RandomNormal


class BANLayer(tf.keras.layers.Layer):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='relu', dropout=0.2, k=3):
        super(BANLayer, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim*2], act=act, dropout=dropout)#FCNet([v_dim, h_dim*k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim*2], act=act, dropout=dropout)#FCNet([q_dim, h_dim*k], act=act, dropout=dropout)

        if k > 1:
            #self.p_net = AveragePooling1D(pool_size=k, strides=1, padding='same')
            self.p_net = AveragePooling2D(pool_size=(self.k, 1), strides=(1, 1), padding='same')
        if h_out > self.c:
            #self.h_mat = self.add_weight(shape=(1, h_out, 1, h_dim*2), initializer=RandomNormal(), trainable=True)#self.h_mat = self.add_weight(shape=(1, h_out, 1, h_dim*k), initializer=RandomNormal(), trainable=True, name='h_mat')
            #self.h_bias = self.add_weight(shape=(1, h_out, 1, 1), initializer=RandomNormal(), trainable=True)
            #self.h_mat = self.add_weight(shape=[1, h_out, 1, h_dim * 2], initializer='random_normal', trainable=True)
            #self.h_bias = self.add_weight(shape=[1, h_out, 1, 1], initializer='random_normal', trainable=True)
            #self.h_mat = tf.Variable(tf.random.normal(shape=(1, h_out, 1, h_dim * 2)))
            #self.h_bias = tf.Variable(tf.random.normal(shape=(1, h_out, 1, 1)))

            self.h_net = Dense(h_out, kernel_initializer='he_normal')

        self.bn = BatchNormalization()
    def build(self, input_shape):
        if self.h_out <= self.c:
            self.h_mat = self.add_weight(
                shape=(1, self.h_out, 1, self.h_dim * 2),
                initializer=tf.compat.v1.glorot_uniform_initializer(),
                trainable=True,
                name='h_mat'
            )
            self.h_bias = self.add_weight(
                shape=(1, self.h_out, 1, 1),
                initializer=tf.compat.v1.glorot_uniform_initializer(),
                trainable=True,
                name='h_bias'
            )
    def attention_pooling(self, v, q, att_map):
        fusion_logits = tf.einsum('bvk,bvq,bqk->bvk', v, att_map, q)#fusion_logits = tf.einsum('bvk,bvq,bqk->bk', v, att_map, q)
        if self.k > 1:
            fusion_logits = tf.expand_dims(fusion_logits, axis=1)
            fusion_logits = self.p_net(fusion_logits)
            fusion_logits = tf.squeeze(fusion_logits, axis=1)
        return fusion_logits

    def call(self, v, q, softmax=False):
        v_num = tf.shape(v)[1]
        q_num = tf.shape(q)[1]

        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = tf.einsum('xhyk,bvk,bqk->bhvq', self.h_mat, v_, q_) + self.h_bias
        else:
            v_ = tf.expand_dims(tf.transpose(self.v_net(v), perm=[0, 2, 1]), axis=3)
            q_ = tf.expand_dims(tf.transpose(self.q_net(q), perm=[0, 2, 1]), axis=2)
            d_ = tf.matmul(v_, q_)
            att_maps = self.h_net(tf.transpose(d_, perm=[0, 2, 3, 1]))
            att_maps = tf.transpose(att_maps, perm=[0, 3, 1, 2])

        if softmax:
            att_maps = tf.nn.softmax(tf.reshape(att_maps, [-1, self.h_out, v_num * q_num]), axis=-1)
            att_maps = tf.reshape(att_maps, [-1, self.h_out, v_num, q_num])

        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps,(self.h_mat,self.h_bias)


class FCNet(tf.keras.layers.Layer):
    def __init__(self, dims, act='relu', dropout=0):
        super(FCNet, self).__init__()
        self.layers_ = []
        for i in range(len(dims) - 1):
            self.layers_.append(Dense(dims[i + 1], activation=act, kernel_initializer='he_normal'))
            if dropout > 0:
                self.layers_.append(Dropout(dropout))

    def call(self, x):
        for layer in self.layers_:
            x = layer(x)
        return x
def rescale_distance_matrix(w): ### For global
    constant_value = tf.constant(1.0,dtype=tf.float32) 
    return (constant_value+tf.math.exp(constant_value))/(constant_value+tf.math.exp(constant_value-w))

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask,adjoin_matrix,dist_matrix):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    if dist_matrix is not None:
        matmul_qk = tf.nn.relu(tf.matmul(q, k, transpose_b = True))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = (tf.multiply(matmul_qk,dist_matrix)) / tf.math.sqrt(dk)
    else:
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads,**kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask,adjoin_matrix,dist_matrix):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask,adjoin_matrix,dist_matrix)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation=gelu),
        keras.layers.Dense(d_model)
    ])
class EncoderLayer(keras.layers.Layer):
    """
    x -> self attention -> add & normalize & dropout
      -> feed_forward -> add & normalize & dropout
    """
    def __init__(self, d_model, num_heads, dff,rate,**kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(int(d_model/2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model/2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon = 1e-6)
        
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.bcn = BANLayer(v_dim=int(d_model / 2), q_dim=int(d_model / 2), h_dim=int(d_model / 2), h_out=2)
    def call(self, x, training, encoder_padding_mask,adjoin_matrix,dist_matrix):
        # x.shape          : (batch_size, seq_len, dim=d_model)
        # attn_output.shape: (batch_size, seq_len, d_model)
        # out1.shape       : (batch_size, seq_len, d_model)
        x1,x2 = tf.split(x,2,-1)
        x_l,attention_weights_local = self.mha1(x1, x1, x1, encoder_padding_mask,adjoin_matrix,dist_matrix = None)
        x_g,attention_weights_global = self.mha2(x2, x2, x2, encoder_padding_mask,adjoin_matrix = None,dist_matrix = dist_matrix)
        attn_output1, att_map, _ = self.bcn(x_l, x_g)
        attn_output2 = tf.concat([x_l, x_g], axis=-1)
        attn_output = tf.multiply(attn_output1, attn_output2)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)
        # ffn_output.shape: (batch_size, seq_len, d_model)
        # out2.shape      : (batch_size, seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)
        # x.shape: (batch_size, input_seq_len, d_model)
        x_l_g = out2
        return x_l_g,attention_weights_local,attention_weights_global

class EncoderModel_motif(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size,
                 d_model, num_heads, dff, rate=0.1,**kwargs):
        super(EncoderModel_motif, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size,
                                                self.d_model)
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)]

    def call(self, x, training,atom_level_features,adjoin_matrix = None,dist_matrix = None):
        encoder_padding_mask = create_padding_mask(x) 
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix[:,tf.newaxis,:,:]
        if dist_matrix is not None:
            dist_matrix = dist_matrix[:,tf.newaxis,:,:]
        # x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
        x = self.dropout(x, training = training)
        x_temp = x[:,1:,:]+ atom_level_features #self.merge_atom_info(x[:,1:,:]+ atom_features)
        x = tf.concat([x[:,0:1,:],x_temp],axis=1)  
        attention_weights_list_local = []
        # xs_local = []
        attention_weights_list_global = []
        for i in range(self.num_layers):
            x,attention_weights_local,attention_weights_global = self.encoder_layers[i](x, training, 
                                       encoder_padding_mask,adjoin_matrix,dist_matrix = dist_matrix) 
            attention_weights_list_local.append(attention_weights_local) 
            attention_weights_list_global.append(attention_weights_global)
        return x,attention_weights_list_local,attention_weights_list_global,encoder_padding_mask

class Co_Attention_Layer(keras.layers.Layer):
    def __init__(self, graph_feat_size, k, num_heads=8,temperature = 0.5, **kwargs):
        self.k = k
        self.graph_feat_size = graph_feat_size
        self.num_heads = num_heads
        self.temperature = temperature
        super(Co_Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化权重矩阵
        self.W_m = self.add_weight(shape=(self.k, self.graph_feat_size),
                                   initializer=tf.compat.v1.glorot_uniform_initializer(),
                                   name='W_m',
                                   trainable=True)
        self.W_v = self.add_weight(shape=(self.k, self.graph_feat_size),
                                   initializer=tf.compat.v1.glorot_uniform_initializer(),
                                   name='W_v',
                                   trainable=True)
        self.W_q = self.add_weight(shape=(self.k, self.graph_feat_size),
                                   initializer=tf.compat.v1.glorot_uniform_initializer(),
                                   name='W_q',
                                   trainable=True)
        self.W_h = self.add_weight(shape=(1, self.k),
                                   initializer=tf.compat.v1.glorot_uniform_initializer(),
                                   name='W_h',
                                   trainable=True)
        super(Co_Attention_Layer, self).build(input_shape)

    def sample_gumbel(self, shape, eps=1e-20):
        uniform_random = tf.random.uniform(shape, minval=0, maxval=1)
        return -tf.math.log(-tf.math.log(uniform_random + eps) + eps)

    def gumbel_softmax(self, logits, temperature):
        gumbel_noise = self.sample_gumbel(tf.shape(logits))
        y = logits + gumbel_noise
        return tf.nn.softmax(y / temperature, axis=-1)

    def call(self, inputs):
        V_n, Q_n = inputs[0], inputs[1]  # 64, 45, 512 | 64, 42, 512

        V_0 = V_n[:, 0, :][:, :, tf.newaxis]  # 64, 512, 1
        Q_0 = Q_n[:, 0, :][:, :, tf.newaxis]  # 64, 512, 1
        V_r = tf.transpose(V_n[:, 1:, :], [0, 2, 1])  # 64, 512, 44
        Q_r = tf.transpose(Q_n[:, 1:, :], [0, 2, 1])  # 64, 512, 41
        M_0 = tf.multiply(V_0, Q_0)  # 64, 512, 1
        H_v = tf.multiply(tf.tanh(tf.matmul(self.W_v, V_r)), tf.tanh(tf.matmul(self.W_m, M_0)))  # 64, k, 44
        H_q = tf.multiply(tf.tanh(tf.matmul(self.W_q, Q_r)), tf.tanh(tf.matmul(self.W_m, M_0)))  # 64, k, 41

        # Compute attention weights with Gumbel-Softmax
        alpha_v = self.gumbel_softmax(tf.matmul(self.W_h, H_v), self.temperature)  # 64, 1, 44
        alpha_q = self.gumbel_softmax(tf.matmul(self.W_h, H_q), self.temperature)  # 64, 1, 41

        vector_v = tf.matmul(alpha_v, tf.transpose(V_r, [0, 2, 1]))  # 64, 1, 512
        vector_q = tf.matmul(alpha_q, tf.transpose(Q_r, [0, 2, 1]))  # 64, 1, 512

        return tf.squeeze(vector_v, 1), tf.squeeze(vector_q, 1), alpha_v, alpha_q





