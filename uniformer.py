#!/usr/bin/python3

from os import environ
environ['KERAS_BACKEND'] = 'jax'
import numpy as np
import keras as K
import keras_cv as KCV

def CBlock(**kwargs):
    # args
    channel = kwargs.get('channel', 768)
    drop_path_rate = kwargs.get('drop_path_rate', 0.)
    mlp_ratio = kwargs.get('mlp_ratio', 4)
    drop_rate = kwargs.get('drop_rate', 0.)
    # network
    inputs = K.Input((None, None, None, channel))
    # positional embedding
    skip = inputs
    pos_embed = K.layers.Conv3D(channel, kernel_size = (3,3,3), padding = 'same')(inputs)
    results = K.layers.Add()([skip, pos_embed])
    # attention
    skip = results
    results = K.layers.BatchNormalization()(results)
    results = K.layers.Conv3D(channel, kernel_size = (1,1,1), padding = 'same')(results)
    results = K.layers.Conv3D(channel, kernel_size = (5,5,5), padding = 'same')(results)
    results = K.layers.Conv3D(channel, kernel_size = (1,1,1), padding = 'same')(results)
    if drop_path_rate > 0:
        results = KCV.layers.DropPath(rate = drop_path_rate)(results)
    else:
        results = K.layers.Identity()(results)
    results = K.layers.Add()([skip, results])
    # mlp
    skip = results
    results = K.layers.BatchNormalization()(results)
    results = K.layers.Conv3D(channel * mlp_ratio, kernel_size = (1,1,1), padding = 'same', activation = K.activations.gelu)(results)
    results = K.layers.Conv3D(channel, kernel_size = (1,1,1), padding = 'same')(results)
    results = K.layers.Dropout(rate = drop_rate)(results)
    if drop_path_rate > 0:
        results = KCV.layers.DropPath(rate = drop_path_rate)(results)
    else:
        results = K.layers.Identity()(results)
    results = K.layers.Add()([skip, results])
    return K.Model(inputs = inputs, outputs = results)

def Attention(**kwargs):
    # args
    channel = kwargs.get('channel', 768)
    num_heads = kwargs.get('num_heads', 8)
    qkv_bias = kwargs.get('qkv_bias', False)
    drop_rate = kwargs.get('drop_rate', 0.3)
    assert channel % num_heads == 0
    # network
    inputs = K.Input((None, channel)) # inputs.shape = (b, s, c)
    results = K.layers.Dense(channel * 3, use_bias = qkv_bias)(inputs) # results.shape = (b, s, 3c)
    results = K.layers.Reshape((-1, 3, num_heads, channel // num_heads))(results) # results.shape = (b, s, 3, h, c // h)
    results = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,3,1,4)))(results) # results.shape = (b, 3, h, s, c // h)
    q, k, v = K.layers.Lambda(lambda x: (x[:,0,...], x[:,1,...], x[:,2,...]))(results) # q.shape = (b, h, s, c // h)
    qk = K.layers.Lambda(lambda x, s: K.ops.matmul(x[0], K.ops.transpose(x[1], (0,1,3,2))) * s, arguments = {'s': (channel // num_heads) ** -0.5})([q, k]) # qk.shape = (b, h, s, s)
    attn = K.layers.Softmax(axis = -1)(qk)
    attn = K.layers.Dropout(rate = drop_rate)(attn)
    qkv = K.layers.Lambda(lambda x: K.ops.transpose(K.ops.matmul(x[0], x[1]), (0, 2, 1, 3)))([attn, v]) # qkv.shape = (b, s, h, c // h)
    results = K.layers.Dense(channel, use_bias = qkv_bias)(qkv)
    results = K.layers.Dropout(rate = drop_rate)(results)
    return K.Model(inputs = inputs, outputs = results)

def SABlock(**kwargs):
    # args
    channel = kwargs.get('channel', 768)
    drop_rate = kwargs.get('drop_rate', 0.3)
    num_heads = kwargs.get('num_heads', 8)
    qkv_bias = kwargs.get('qkv_bias', False)
    drop_path_rate = kwargs.get('drop_path_rate', 0.)
    # network
    inputs = K.Input((None, None, None, channel))
    # positional embedding
    skip = inputs
    pos_embed = K.layers.Conv3D(channel, kernel_size = (3,3,3), padding = 'same')(inputs)
    results = K.layers.Add()([skip, pos_embed])
    # attention between h and w
    skip = results
    results = K.layers.LayerNormalization()(results) # results.shape = (batch, T, H, W, channel)
    shape1 = K.layers.Lambda(lambda x: K.ops.shape(x))(results)
    results = K.layers.Lambda(lambda x: K.ops.reshape(x, (K.ops.shape(x)[0] * K.ops.shape(x)[1],
                                                          K.ops.shape(x)[2] * K.ops.shape(x)[3],
                                                          K.ops.shape(x)[4])))(results) # results.shape = (batch * T, H * W, channel)
    results = Attention(**kwargs)(results)
    results = KCV.layers.DropPath(rate = drop_path_rate)(results)
    results = K.layers.Lambda(lambda x: K.ops.reshape(x[0], x[1]))([results, shape1]) # results.shape = (batch, T, H, W, channel)
    results = K.layers.Add()([skip, results])
    # attention between t and h
    skip = results
    results = K.layers.LayerNormalization()(results) # results.shape = (batch, T, H, W, channel)
    results = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,3,1,2,4)))(results) # results.shape = (batch, W, T, H, channel)
    shape2 = K.layers.Lambda(lambda x: K.ops.shape(x))(results)
    results = K.layers.Lambda(lambda x: K.ops.reshape(x, (K.ops.shape(x)[0] * K.ops.shape(x)[1],
                                                          K.ops.shape(x)[2] * K.ops.shape(x)[3],
                                                          K.ops.shape(x)[4])))(results) # results.shape = (batch * W, T * H, channel)
    results = Attention(**kwargs)(results)
    results = KCV.layers.DropPath(rate = drop_path_rate)(results)
    results = K.layers.Lambda(lambda x: K.ops.reshape(x[0], x[1]))([results, shape2]) # results.shape = (batch, W, T, H, channel)
    results = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,3,1,4)))(results) # results.shape = (batch, T, H, W, channel)
    results = K.layers.Add()([skip, results])
    # attention between w and t
    skip = results
    results = K.layers.LayerNormalization()(results) # results.shape = (batch, T, H, W, channel)
    results = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,2,3,1,4)))(results) # results.shape = (batch, H, W, T, channel)
    shape3 = K.layers.Lambda(lambda x: K.ops.shape(x))(results)
    results = K.layers.Lambda(lambda x: K.ops.reshape(x, (K.ops.shape(x)[0] * K.ops.shape(x)[1],
                                                          K.ops.shape(x)[2] * K.ops.shape(x)[3],
                                                          K.ops.shape(x)[4])))(results) # results.shape = (batch * H, W * T, channel)
    results = Attention(**kwargs)(results)
    results = KCV.layers.DropPath(rate = drop_path_rate)(results)
    results = K.layers.Lambda(lambda x: K.ops.reshape(x[0], x[1]))([results, shape3]) # results.shape = (batch, H, W, T, channel)
    results = K.layers.Lambda(lambda x: K.ops.transpose(x, (0,3,1,2,4)))(results) # results.shape = (batch, T, H, W, channel)
    results = K.layers.Add()([skip, results])
    return K.Model(inputs = inputs, outputs = results)

def Uniformer(**kwargs):
    # args
    in_channel = kwargs.get('in_channel', 3)
    out_channel = kwargs.get('out_channel', None)
    hidden_channels = kwargs.get('hidden_channels', [64,128,320,512])
    depth = kwargs.get('depth', [5,8,20,7])
    mlp_ratio = kwargs.get('mlp_ratio', 4.)
    drop_rate = kwargs.get('drop_rate', 0.3)
    global_drop_path_rate = kwargs.get('global_drop_path_rate', 0.)
    qkv_bias = kwargs.get('qkv_bias', False)
    num_heads = kwargs.get('num_heads', 8)
    assert len(hidden_channels) == len(depth)
    dpr = [x.item() for x in np.linspace(0, global_drop_path_rate, sum(depth))]
    # network
    inputs = K.Input((None, None, None, in_channel)) # inputs.shape = (batch, t, h, w, in_channel)
    results = K.layers.Conv3D(hidden_channels[0], kernel_size = (4, 4, 4), strides = (4, 4, 4), padding = 'same')(inputs) # results.shape = (batch, t / 4, h / 4, w / 4, hidden_channels[0])
    results = K.layers.LayerNormalization()(results)
    results = K.layers.Dropout(rate = drop_rate)(results)
    # block 1
    for i in range(depth[0]):
        results = CBlock(channel = hidden_channels[0], drop_path_rate = dpr[i], **kwargs)(results)
    results = K.layers.Conv3D(hidden_channels[1], kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = 'same')(results) # results.shape = (batch, t / 8, h / 8, w / 8, hidden_channels[1])
    results = K.layers.LayerNormalization()(results)
    # block 2
    for i in range(depth[1]):
        results = CBlock(channel = hidden_channels[1], drop_path_rate = dpr[i], **kwargs)(results)
    results = K.layers.Conv3D(hidden_channels[2], kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = 'same')(results) # results.shape = (batch, t / 16, h / 16, w / 16, hidden_channels[2])
    results = K.layers.LayerNormalization()(results)
    # do attention only when the feature shape is small enough
    # block 3
    for i in range(depth[2]):
        results = SABlock(channel = hidden_channels[2], drop_path_rate = dpr[i], qkv_bias = qkv_bias, num_heads = num_heads, **kwargs)(results)
    results = K.layers.Conv3D(hidden_channels[3], kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = 'same')(results) # results.shape = (batch, t / 32, h / 32, w / 32, hidden_channels[3])
    results = K.layers.LayerNormalization()(results)
    # block 4
    for i in range(depth[3]):
        results = SABlock(channel = hidden_channels[3], drop_path_rate = dpr[i], qkv_bias = qkv_bias, num_heads = num_heads, **kwargs)(results)
    results = K.layers.BatchNormalization()(results) # results.shape = (batch, t / 32, h / 32, w / 32, hidden_channels[3])
    if out_channel is not None:
        results = K.layers.Dense(out_channel, activation = K.activations.tanh)(results) # results.shape = (batch, t / 32, h / 32, w / 32, out_channel)
    results = K.layers.Lambda(lambda x: K.ops.mean(x, axis = (1,2,3)))(results) # results.shape = (batch, out_channel)
    return K.Model(inputs = inputs, outputs = results)

if __name__ == "__main__":
    inputs = np.random.normal(size = (1,4,4,4,768))
    sablock = SABlock(channel = 768, drop_path_rate = 0, qkv_bias = False, num_heads = 8)
    sablock.save('sablock.keras')
    results = sablock(inputs)
    print(results.shape)
    inputs = np.random.normal(size = (1,64,64,64,3))
    uniformer = Uniformer(in_channel = 3, out_channel = 100)
    uniformer.save('uniformer.keras')
    outputs = uniformer(inputs = inputs)
    print(outputs.shape)
