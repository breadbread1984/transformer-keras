#!/usr/bin/python3

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
import numpy as np
import keras as K

def Attention(hidden_size, head_num, is_causal = False):
  inputs = K.Input((None, hidden_size)) # inputs.shape = (batch, seq_len, hidden_size)
  qkv = K.layers.Dense(3 * hidden_size, use_bias = False)(inputs) # qkv.shape = (batch, seq_len, 3 * hidden_size)
  q, k, v = K.layers.Lambda(lambda x, d: K.ops.split(x, [d,2*d], axis = -1), arguments = {'d': hidden_size})(qkv)
  q = K.layers.Reshape((-1, head_num, hidden_size // head_num))(q) # q.shape = (batch, seq_len, head_num, hidden_size // head_num)
  k = K.layers.Reshape((-1, head_num, hidden_size // head_num))(k) # k.shape = (batch, seq_len, head_num, hidden_size // head_num)
  v = K.layers.Reshape((-1, head_num, hidden_size // head_num))(v) # v.shape = (batch, seq_len, head_num, hidden_size // head_num)
  q = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,1,3)))(q) # q.shape = (batch, head_num, seq_len, hidden_size // head_num)
  k = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,1,3)))(k) # k.shape = (batch, head_num, seq_len, hidden_Size // head_num)
  v = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,1,3)))(v) # v.shape = (batch, head_num, seq_len, hidden_size // head_num)
  if is_causal:
    mask = K.layers.Lambda(lambda x: (K.ops.expand_dims(K.ops.arange(K.ops.shape(x[1])[2]) + 1, axis = 1) > K.ops.expand_dims(K.ops.arange(K.ops.shape(x[1])[2]), axis = 0))[-K.ops.shape(x[0])[2]:,:])([q,k]) # mask.shape = (q_seq_len, k_seq_len)
  else:
    mask = K.layers.Lambda(lambda x: K.ops.ones((K.ops.shape(x[0])[2], K.ops.shape(x[1])[2])))([q, k]) # mask.shape = (q_seq_len, k_seq_len)
  qk = K.layers.Lambda(lambda x: K.ops.matmul(x[0], K.ops.transpose(x[1], (0,1,3,2))))([q, k]) # qk.shape = (batch, head_num, q_seq_len, k_seq_len)
  qk = K.layers.Lambda(lambda x, n_inf: K.ops.where(K.ops.expand_dims(K.ops.expand_dims(x[1], axis = 0), axis = 0), # mask.shape = (1, 1, q_seq_len,k_seq_len)
                                                    x[0],
                                                    n_inf),
                       arguments = {'n_inf': np.finfo(np.float32).min})([qk,mask])
  qk = K.layers.Lambda(lambda x, d: K.ops.softmax(x[0] / K.ops.sqrt(d), axis = -1), arguments = {'d': hidden_size // head_num})(qk)
  qkv = K.layers.Lambda(lambda x: K.ops.matmul(x[0], x[1]))([qk, v])
  qkv = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,1,3)))(qkv) # qkv.shape = (batch, seq_len, head_num, hidden_size // head_num)
  qkv = K.layers.Reshape((-1, hidden_size))(qkv) # qkv.shape = (batch, seq_len, hidden_size)
  qkv = K.layers.Dense(hidden_size, use_bias = False)(qkv) # qkv.shape = (batch, seq_len, hidden_size)
  return K.Model(inputs = inputs, outputs = qkv)

if __name__ == "__main__":
  attn = Attention(128, 8, True)
  inputs = np.random.normal(size = (2, 12, 128))
  output = attn(inputs)
  print(output)
