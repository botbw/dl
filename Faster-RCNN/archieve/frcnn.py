import imp
from operator import imod
import os
import sys

import tensorflow as tf
from config import cfg
from rpn import rpn

import numpy as np

class frcnn(tf.keras.Model):
  def __init__(self, name = "frcnn"):
    super().__init__(name)
    self.backbone = cfg.frcnn.backbone
    self.proposal = rpn()

  def call(self, img, training = False, gt_boxes = None):
    img_sz = img.shape[1:2]
    backbone_output = self.backbone(img)

