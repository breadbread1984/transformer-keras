#!/usr/bin/python3

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
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
    skip = inputs
    pos_embed = K.layers.Conv3D(channel, kernel_size = (3,3,3), padding = 'same')(inputs)
    results = K.layers.Add()([skip, pos_embed])

    skip = results
    results = K.layers.BatchNormalization()(results)
    results = K.layers.Conv3D(channel, kernel_size = (1,1,1), padding = 'same')(results)
    results = K.layers.Conv3D(channel, kernel_size = (5,5,5), strides = (2,2,2), padding = 'same')(results)
    results = K.layers.Conv3D(channel, kernel_size = (1,1,1), padding = 'same')(results)
    if drop_path_rate > 0:
        results = KCV.layers.DropPath(rate = drop_path_rate)(results)
    else:
        results = K.layers.Identity()(results)
    results = K.layers.Add()([skip, results])

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

def Uniformer(**kwargs):
    # args
    in_channel = kwargs.get('in_channel', 3)
    hidden_channels = kwargs.get('hidden_channels', [64,128,320,512])
    depth = kwargs.get('depth', [5,8,20,7])
    mlp_ratio = kwargs.get('mlp_ratio', 4.)
    drop_rate = kwargs.get('drop_rate', 0.3)
    global_drop_path_rate = kwargs.get('global_drop_path_rate', 0.)
    assert len(hidden_channels) == len(depth)
    dpr = [x.item() for x in np.linspace(0, global_drop_path_rate, sum(depth))]
    # network
    inputs = K.Input((None, None, None, in_channel)) # inputs.shape = (batch, t, h, w, in_channel)
    results = K.layers.Conv3D(hidden_channels[0], kernel_size = (patch_size, patch_size, patch_size), strides = (patch_size, patch_size, patch_size), padding = 'same')(inputs) # results.shape = (batch, t / patch_size, h / patch_size, w / patch_size, out_channel)
    results = K.layers.LayerNormalization()(results)
    results = K.layers.Dropout(rate = drop_rate)(results)
    for i in range(depth[0]):
        results = CBlock(channel = hidden_channel[0], drop_path_rate = dpr[i], **kwargs)(results)
    
