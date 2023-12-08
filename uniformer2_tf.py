#!/usr/bin/python3

import numpy as np
import tensorflow as tf
import keras_cv

model = keras_cv.models.ViTDetBackbone.from_preset("vitdet_base_sa1b")
model.save('vit_b16.keras')
