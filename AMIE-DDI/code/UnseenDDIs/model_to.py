import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
        # 在 build 方法中初始化 h_mat 和 h_bias
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
        return logits, att_maps


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

def rescale_distance_matrix(w):  ### For global
    constant_value = tf.constant(1.0, dtype=tf.float32)
    return (constant_value + tf.math.exp(constant_value)) / (constant_value + tf.math.exp(constant_value - w))


def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.)))


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
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


def create_padding_mask_atom(batch_data):
    padding_mask = tf.cast(tf.math.equal(tf.reduce_sum(batch_data, axis=-1), 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]
def create_padding_mask(batch_data):
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
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
        matmul_qk = tf.nn.relu(tf.matmul(q, k, transpose_b=True))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = (tf.multiply(matmul_qk, dist_matrix)) / tf.math.sqrt(dk)
    else:
        matmul_qk = tf.matmul(q, k, transpose_b=True)

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
    def __init__(self, d_model, num_heads, **kwargs):
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

    def call(self, q, k, v, mask, adjoin_matrix, dist_matrix):
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
            q, k, v, mask, adjoin_matrix, dist_matrix)
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

    def __init__(self, d_model, num_heads, dff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.mha1 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.mha2 = MultiHeadAttention(int(d_model / 2), num_heads)
        self.ffn = feed_forward_network(d_model, dff)
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.bcn = BANLayer(v_dim=int(d_model/2), q_dim=int(d_model/2), h_dim=int(d_model/2), h_out=2)

    def call(self, x1, x2, training, encoder_padding_mask1,encoder_padding_mask2, adjoin_matrix1,adjoin_matrix2, dist_matrix1,dist_matrix2):
        x11, x12 = tf.split(x1, 2, -1)
        x21, x22 = tf.split(x2, 2, -1)
        x_l1, attention_weights_local1 = self.mha1(x11, x21, x11, encoder_padding_mask1, adjoin_matrix1, dist_matrix=None)
        x_g1, attention_weights_global1 = self.mha2(x12, x22, x12, encoder_padding_mask1, adjoin_matrix=None, dist_matrix=dist_matrix1)
        x_l2, attention_weights_local2 = self.mha1(x21, x11, x21, encoder_padding_mask2, adjoin_matrix2, dist_matrix=None)
        x_g2, attention_weights_global2 = self.mha2(x22, x12, x22, encoder_padding_mask2, adjoin_matrix=None, dist_matrix=dist_matrix2)
        attn_output1, att_map1 = self.bcn(x_l1, x_g1)
        attn_output2, att_map2 = self.bcn(x_l2, x_g2)
        #x_l1, att_l1 = self.bcn(x_l1, x_l2)
        #x_g1, att_g1 = self.bcn(x_g1, x_g2)
        #x_l2, att_l2 = self.bcn(x_l2, x_l1)
        #x_g2, att_g2 = self.bcn(x_g2, x_g1)
        #attn_output1 = tf.concat([x_l1, x_g1], axis=-1)
        #attn_output2 = tf.concat([x_l2, x_g2], axis=-1)
        #x, map = self.bcn(attn_output1, attn_output2)
        attn_output1 = self.dropout1(attn_output1, training=training)
        attn_output2 = self.dropout1(attn_output2, training=training)
        out11 = self.layer_norm1(x1 + attn_output1)
        out21 = self.layer_norm1(x2 + attn_output2)
        ffn_output1 = self.ffn(out11)
        ffn_output2 = self.ffn(out21)
        ffn_output1 = self.dropout2(ffn_output1, training=training)
        ffn_output2 = self.dropout2(ffn_output2, training=training)
        out11 = self.layer_norm2(out11 + ffn_output1)
        out21 = self.layer_norm2(out21 + ffn_output2)
        #x, map = self.bcn(out21, out22)
        return out11,out21, attention_weights_local1, attention_weights_global1, attention_weights_local2, attention_weights_global2


class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers,input_vocab_size,
                 d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        # self.max_length = max_length

        self.embedding = keras.layers.Dense(self.d_model, activation='relu')
        self.embedding_motif = keras.layers.Embedding(input_vocab_size,
                                                self.d_model*2)
        self.dff = dff
        # position_embedding.shape: (1, max_length, d_model)
        # self.position_embedding = get_position_embedding(max_length,
        #                                                  self.d_model)
        self.global_embedding = keras.layers.Dense(dff, activation='relu')

        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(int(d_model), num_heads, dff, rate)
            for _ in range(self.num_layers)]
        self.encoder_layers_m = [
            EncoderLayer(int(d_model*2), num_heads, dff, rate)
            for _ in range(self.num_layers)]
        self.bcn = BANLayer(v_dim=self.d_model, q_dim=self.d_model, h_dim=self.d_model, h_out=2)
        #self.bcn1D = BANLayer1D(v_dim=self.d_model, q_dim=self.d_model, h_dim=self.d_model, h_out=2)
    def call(self, x1, x2, x_m1, x_m2, training, adjoin_matrix_atom1=None, adjoin_matrix_atom2=None,
             dist_matrix_atom1=None, dist_matrix_atom2=None, atom_match_matrix1=None, atom_match_matrix2=None, sum_atom1=None, sum_atom2=None, adjoin_matrix_motif1=None, adjoin_matrix_motif2=None, dist_matrix_motif1=None, dist_matrix_motif2=None):
        # x.shape: (batch_size, input_seq_len)
        #input_seq_len = tf.shape(x)[1]

        encoder_padding_mask_atom1 = create_padding_mask_atom(x1)
        encoder_padding_mask_atom2 = create_padding_mask_atom(x2)
        encoder_padding_mask_motif1 = create_padding_mask(x_m1)
        encoder_padding_mask_motif2 = create_padding_mask(x_m2)
        if adjoin_matrix_atom1 is not None:
            adjoin_matrix_atom1 = adjoin_matrix_atom1[:, tf.newaxis, :, :]
        if adjoin_matrix_atom2 is not None:
            adjoin_matrix_atom2 = adjoin_matrix_atom2[:, tf.newaxis, :, :]
        if dist_matrix_atom1 is not None:
            dist_matrix_atom1 = dist_matrix_atom1[:, tf.newaxis, :, :]
        if dist_matrix_atom2 is not None:
            dist_matrix_atom2 = dist_matrix_atom2[:, tf.newaxis, :, :]
        if adjoin_matrix_motif1 is not None:
            adjoin_matrix_motif1 = adjoin_matrix_motif1[:,tf.newaxis,:,:]
        if adjoin_matrix_motif2 is not None:
            adjoin_matrix_motif2 = adjoin_matrix_motif2[:,tf.newaxis,:,:]
        if dist_matrix_motif1 is not None:
            dist_matrix_motif1 = dist_matrix_motif1[:,tf.newaxis,:,:]
        if dist_matrix_motif2 is not None:
            dist_matrix_motif2 = dist_matrix_motif2[:,tf.newaxis,:,:]
        # x.shape: (batch_size, input_seq_len, d_model)
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x1 = self.dropout(x1, training=training)
        x2 = self.dropout(x2, training=training)
        attention_weights_list_local_atom1 = []
        attention_weights_list_global_atom1 = []
        attention_weights_list_local_atom2 = []
        attention_weights_list_global_atom2 = []

        for i in range(self.num_layers):
            x1,x2, attention_weights_local_atom1, attention_weights_global_atom1, attention_weights_local_atom2, attention_weights_global_atom2 = self.encoder_layers[i](x1,x2, training, encoder_padding_mask_atom1,encoder_padding_mask_atom2, adjoin_matrix_atom1,adjoin_matrix_atom2, dist_matrix_atom1, dist_matrix_atom2)
            attention_weights_list_local_atom1.append(attention_weights_local_atom1)
            attention_weights_list_global_atom1.append(attention_weights_global_atom1)
            attention_weights_list_local_atom2.append(attention_weights_local_atom2)
            attention_weights_list_global_atom2.append(attention_weights_global_atom2)

        x1 = tf.matmul(atom_match_matrix1, x1) / sum_atom1
        x1 = self.global_embedding(x1)
        x2 = tf.matmul(atom_match_matrix2, x2) / sum_atom2
        x2 = self.global_embedding(x2)
        #x, att_x, _ = self.bcn(x1, x2)
        x_m1 = self.embedding_motif(x_m1)
        x_m2 = self.embedding_motif(x_m2)
        x_m1 *= tf.math.sqrt(tf.cast(self.d_model*2, tf.float32))
        x_m2 *= tf.math.sqrt(tf.cast(self.d_model * 2, tf.float32))
        x_m1 = self.dropout(x_m1, training=training)
        x_m2 = self.dropout(x_m2, training=training)
        x_temp1 = x_m1[:, 1:, :] + x1  # self.merge_atom_info(x[:,1:,:]+ atom_features)
        x_temp2 = x_m2[:, 1:, :] + x2
        x_m1 = tf.concat([x_m1[:, 0:1, :], x_temp1], axis=1)
        x_m2 = tf.concat([x_m2[:, 0:1, :], x_temp2], axis=1)
        attention_weights_list_local_motif1 = []
        # xs_local = []
        attention_weights_list_global_motif1 = []
        attention_weights_list_local_motif2 = []
        # xs_local = []
        attention_weights_list_global_motif2 = []
        for i in range(self.num_layers):
            x_m1,x_m2, attention_weights_local_motif1, attention_weights_global_motif1, attention_weights_local_motif2, attention_weights_global_motif2 = self.encoder_layers_m[i](x_m1,x_m2, training,
                                                                                          encoder_padding_mask_motif1,encoder_padding_mask_motif2,
                                                                                          adjoin_matrix_motif1,adjoin_matrix_motif2,
                                                                                          dist_matrix_motif1,dist_matrix_motif2)
            attention_weights_list_local_motif1.append(attention_weights_local_motif1)
            attention_weights_list_global_motif1.append(attention_weights_global_motif1)
            attention_weights_list_local_motif2.append(attention_weights_local_motif2)
            attention_weights_list_global_motif2.append(attention_weights_global_motif2)

        #x_m, att_m, _ = self.bcn1D(x_m1, x_m2)
        #final = tf.concat([x, x_m], axis=1)
        #f = tf.reshape(f,[tf.shape(f)[0],1,tf.shape(f)[1]])
        return x_m1, x_m2, attention_weights_list_local_atom1, attention_weights_list_global_atom1, attention_weights_list_local_atom2, attention_weights_list_global_atom2,attention_weights_list_local_motif1,attention_weights_list_global_motif1, attention_weights_list_local_motif2,attention_weights_list_global_motif2, encoder_padding_mask_atom1,encoder_padding_mask_motif1, encoder_padding_mask_atom2,encoder_padding_mask_motif2



