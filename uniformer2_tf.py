#!/usr/bin/python3

import numpy as np
import tensorflow as tf
from vit_keras import vit

model = vit.vit_b16(image_size = 384, activation = 'sigmoid', pretrained = True, include_top = False, pretrained_top = False)
model.save('vit_b16.keras')
