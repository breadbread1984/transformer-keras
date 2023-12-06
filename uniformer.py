#!/usr/bin/python3

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
import keras as K

def CBlock(**kwargs):
    # args
    in_channel = kwargs.get('in_channel', 768)
    # network
    inputs = K.Input((None, None, None, in_channel))
    

def Uniformer(**kwargs):
    # args
    in_channel = kwargs.get('in_channel', 3)
    out_channel = kwargs.get('out_channel', 768)
    drop_rate = kwargs.get('drop_rate', 0.3)
    # network
    inputs = K.Input((None, None, None, in_channel)) # inputs.shape = (batch, t, h, w, in_channel)
    results = K.layers.Conv3D(out_channel, kernel_size = (patch_size, patch_size, patch_size), strides = (patch_size, patch_size, patch_size), padding = 'same')(inputs) # results.shape = (batch, t / patch_size, h / patch_size, w / patch_size, out_channel)
    results = K.layers.LayerNorm()(results)
    results = K.layers.Dropout(rate = drop_rate)(results)
    