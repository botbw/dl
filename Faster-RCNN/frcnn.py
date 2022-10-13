import imp
from operator import imod
import os
import sys

import tensorflow as tf
from anchor import generate_anchors
from config import cfg
from rpn import rpn

import numpy as np

class frcnn(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.backbone = cfg.frcnn.backbone
    self.rpn_conv = tf.keras.layers.Conv2D(cfg.rpn_chan, kernel_size=(3, 3), padding='same', activation='relu', use_bias=cfg.frcnn.use_bias, kernel_initializer=cfg.frcnn.initializer, kernel_regularizer=cfg.frcnn.regulizer)
    self.proposal = rpn()

  def call(self, img, training=False):
    img_sz = img.shape[1:2]
    backbone_output = self.backbone(img)
    rpn_feature = self.rpn_conv(backbone_output)

    anchors = generate_anchors(rpn_feature.shape[0], rpn_feature.shape[1])
