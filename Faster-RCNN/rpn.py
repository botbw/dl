from lib2to3.pytree import convert
import tensorflow as tf
import numpy as np
from config import cfg

class rpn(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.rpn_cls_score_conv = tf.keras.layers.Conv2D(cfg.anchors.num * 2, kernel_size=(1, 1), padding='same', use_bias=cfg.frcnn.use_bias, kernel_initializer=cfg.frcnn.initializer, kernel_regularizer=cfg.frcnn.regulizer)
    self.rpn_cls_prob_softmax = tf.keras.layers.Softmax()
    self.rpn_bbox_pred_conv = tf.keras.layers.Conv2D(cfg.anchors.num * 4,kernel_size=(1, 1), padding='same',  use_bias=cfg.frcnn.use_bias, kernel_initializer=cfg.frcnn.initializer, kernel_regularizer=cfg.frcnn.regulizer)

  def call(self, rpn_feature, anchors, img_sz, is_training = False):
    rpn_cls_score = self.rpn_cls_score_conv(rpn_feature)
    rpn_cls_score_reshape = tf.reshape(rpn_cls_score, (-1, 2))
    rpn_cls_prob = self.rpn_cls_prob_softmax(rpn_cls_score_reshape)
    rpn_cls_pred = tf.argmax(rpn_cls_prob, axis = -1)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, tf.shape(rpn_cls_score))

    rpn_bbox_pred = self.rpn_bbox_pred_conv(rpn_feature)
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (-1, 4))

    rpn_cls_prob_fgrd = rpn_cls_prob[:, :, :, cfg.anchors.num:]
    rpn_cls_prob_fgrd = tf.reshape(rpn_cls_prob_fgrd, (-1, ))

    anchor_preds = self.convert_and_clip(anchors, rpn_bbox_pred, img_sz)

    proposals = tf.image.non_max_suppression(anchor_preds, rpn_cls_prob_fgrd, max_output_size=cfg.frcnn.nms.output_size, iou_threshold=cfg.frcnn.nms.iou)

  def convert_and_clip(self, anchors, pred, img_sz):
    # widths of anchors (y_high - y_low)
    w = anchors[:, 3] - anchors[:, 1] + 1
    # hights of anchors
    h = anchors[:, 2] - anchors[:, 0] + 1
    # centre of anchors
    x = anchors[: 0] + 0.5 * h
    y = anchors[: 1] + 0.5 * w
    # predict values
    tx = pred[:, 0]
    ty = pred[:, 1]
    tw = pred[:, 2]
    th = pred[:, 3]
    # convert predicted values to bounding boxes
    x_pred = tx * h + x
    y_pred = ty * w + y
    w_pred = tf.exp(tw) * w
    h_pred = tf.exp(th) * h
    # convert predicted bounding boxes to (x0, y0, x1, y1)
    x0 = x_pred - 0.5 * h_pred
    x1 = x_pred + 0.5 * h_pred
    y0 = y_pred - 0.5 * y_pred
    y1 = y_pred + 0.5 * y_pred

    x0 = tf.maximum(x0, 0)
    x1 = tf.minimum(x1, img_sz[0])
    y0 = tf.maximum(y0, 0)
    y1 = tf.minimum(y1, img_sz[1])

    return tf.stack([x0, y0, x1, y1], axis = 1)




    
