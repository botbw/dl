from sys import _AsyncgenHook
from webbrowser import get
import tensorflow as tf
import numpy as np
from config import cfg
from anchor import generate_anchors


class proposal_layer(tf.keras.layers.Layer):
  def __init__(self, name = "proposal"):
    super().__init__(name)
    self.rpn_cls_score_conv = tf.keras.layers.Conv2D(cfg.anchors.num * 2, kernel_size=(1, 1), padding='same', use_bias=cfg.frcnn.use_bias, kernel_initializer=cfg.frcnn.initializer, kernel_regularizer=cfg.frcnn.regulizer)
    self.rpn_cls_prob_softmax = tf.keras.layers.Softmax()
    self.rpn_bbox_pred_conv = tf.keras.layers.Conv2D(cfg.anchors.num * 4,kernel_size=(1, 1), padding='same',  use_bias=cfg.frcnn.use_bias, kernel_initializer=cfg.frcnn.initializer, kernel_regularizer=cfg.frcnn.regulizer)

  def call(self, rpn_features, anchors, img_sz, training):
    if training:
      max_output_size = cfg.train.nms.max_output_size
      iou_threshold = cfg.train.nms.iou_threshold
    else:
      max_output_size = cfg.test.nms.max_output_size
      iou_threshold = cfg.test.nms.iou_threshold

    rpn_cls_score = self.rpn_cls_score_conv(rpn_features)
    rpn_cls_score_reshape = tf.reshape(rpn_cls_score, (-1, 2))
    rpn_cls_prob = self.rpn_cls_prob_softmax(rpn_cls_score_reshape)
    # rpn_cls_pred = tf.argmax(rpn_cls_prob, axis = -1)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, tf.shape(rpn_cls_score))

    rpn_bbox_pred = self.rpn_bbox_pred_conv(rpn_features)
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (-1, 4))

    rpn_cls_prob_fgrd = rpn_cls_prob[:, :, :, cfg.anchors.num:]
    rpn_cls_prob_fgrd = tf.reshape(rpn_cls_prob_fgrd, (-1, ))

    anchor_preds = self.transform_inv_and_clip(anchors, rpn_bbox_pred, img_sz)
    # keep = _filter_boxes(proposals, min_size * im_info[2])

    proposal_ids = tf.image.non_max_suppression(anchor_preds, rpn_cls_prob_fgrd, max_output_size=max_output_size, iou_threshold=iou_threshold)

    proposals = anchor_preds[proposal_ids]
    proposal_scores = rpn_cls_prob_fgrd[proposal_ids]

    return proposals, proposal_scores

  def transform_inv_and_clip(self, anchors, pred, img_sz):
    t = bbox_transform_inv(anchors, pred)
    x0 = t[:, 0]
    y0 = t[:, 1]
    x1 = t[:, 2]
    y1 = t[:, 3]

    x0 = tf.maximum(x0, 0)
    x1 = tf.minimum(x1, img_sz[0])
    y0 = tf.maximum(y0, 0)
    y1 = tf.minimum(y1, img_sz[1])

    boxes = tf.stack([x0, y0, x1, y1], axis = 1)
    return boxes

class anchor_target_layer(tf.keras.layers.Layer): # only in training
  def __init__(self, name = "roi-data"):
    super().__init__(name)

  def random_disable_labels(self, labels_input, inds, disable_nums):
      shuffle_fg_inds = tf.random.shuffle(inds)
      disable_inds = shuffle_fg_inds[:disable_nums]
      disable_inds_expand_dim = tf.expand_dims(disable_inds, axis=1)
      neg_ones = tf.ones_like(disable_inds, dtype=tf.float32) * -1.
      return tf.tensor_scatter_nd_update(labels_input, disable_inds_expand_dim, neg_ones)

  def unmap(self, data, count, inds, fill, type):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if type == 'labels':
        ret = tf.fill((count,), value = fill, dtype=tf.float32, name="unmap_" + type)
        inds_expand = tf.expand_dims(inds, axis=1)
        return tf.tensor_scatter_nd_update(ret, inds_expand, data)
    else:
        ret = tf.fill(tf.concat([[count, ], tf.shape(data)[1:]], axis=0), value = fill,dtype=tf.float32, name="unmap_" + type)
        inds_expand = tf.expand_dims(inds, axis=1)
        return tf.tensor_scatter_nd_update(ret, inds_expand, data)

  def call(self, anchors, img_sz, gt_boxes): 
    ori_anchors_num = anchors.shape[0]

    mask = ((anchors[:, 0] >= -0.0) &
            (anchors[:, 1] >= -0.0) &
            (anchors[:, 2] < (img_sz[1] + 0.0)) &  # width
            (anchors[:, 3] < (img_sz[0] + 0.0)))  # height
    inds_inside = tf.reshape(tf.where(mask), shape=(-1,))
    anchors = anchors[mask]
    labels = tf.zeros(anchors.shape[0], dtype = 'float')
    labels -= 1 # dont care

    overlaps = get_overlap(anchors, gt_boxes)
    
    argmax_overlaps = tf.cast(tf.argmax(overlaps, axis=1), dtype=tf.int32)
    argmax_gather_nd_inds = tf.stack([tf.range(tf.shape(overlaps)[0]), argmax_overlaps], axis=1)
    max_overlaps = tf.gather_nd(overlaps, argmax_gather_nd_inds)

    gt_argmax_overlaps = tf.cast(tf.argmax(overlaps, axis=0), dtype=tf.int32)
    max_overlaps_gather_nd_inds = tf.stack([gt_argmax_overlaps, tf.range(tf.shape(overlaps)[1])], axis=1)
    gt_max_overlaps = tf.gather_nd(overlaps, max_overlaps_gather_nd_inds)
    gt_argmax_overlaps = tf.where(overlaps == gt_max_overlaps)[:, 0]
    # (ii)
    labels = tf.where(max_overlaps < cfg.train.rpn_negative_threshold, tf.zeros_like(labels, dtype = 'float'), labels)
    labels = tf.where(max_overlaps >= cfg.train.rpn_positive_threshold, tf.ones_like(labels, dtype = 'float'), labels)
    #(i)
    unique_gt_argmax_overlaps = tf.unique(gt_argmax_overlaps)[0]
    highest_fg_label = tf.ones(unique_gt_argmax_overlaps.shape)
    highest_gt_row_ids_expand_dim = tf.expand_dims(unique_gt_argmax_overlaps, axis=1)
    labels = tf.tensor_scatter_nd_update(labels, highest_gt_row_ids_expand_dim, highest_fg_label)
    
    # Instead, we randomly sample 256 anchors in an image to
    # compute the loss function of a mini-batch, where the 
    # sampled positive and negative anchors have a ratio of up to 1:1. 
    # If there are fewer than 128 positive samples in an image, 
    # we pad the mini-batch with negative ones.
 
    # subsample positive labels if we have too many  
    num_fg = int(cfg.train.batch_size * cfg.train.batch_size)
    fg_inds = tf.reshape(tf.where(labels == 1), shape=(-1,))
    fg_inds_num = tf.shape(fg_inds)[0]

    fg_flag = tf.cast(fg_inds_num > num_fg, dtype=tf.float32)
    labels = fg_flag * self.random_disable_labels(labels, fg_inds, fg_inds_num - num_fg) + \
              (1.0 - fg_flag) * labels

    # subsample negative labels if we have too many
    num_bg = cfg.train.batch_size - tf.shape(tf.where(labels == 1))[0]
    # bg_inds = np.where(labels == 0)[0]
    bg_inds = tf.reshape(tf.where(labels == 0), shape=(-1,))
    bg_inds_num = tf.shape(bg_inds)[0]
    bg_flag = tf.cast(bg_inds_num > num_bg, dtype=tf.float32)
    labels = bg_flag * self.random_disable_labels(labels, bg_inds, bg_inds_num - num_bg) + \
              (1.0 - bg_flag) * labels

    # 此处将每个anchor与gt_box对准，gt_box与anchor的dx,dy,dw,dh，用来与模型预测的box计算损失
    bbox_targets = bbox_transform_tf(anchors, tf.gather(gt_boxes, argmax_overlaps, axis=0)[:, :4])

    # bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights = tf.zeros((tf.shape(inds_inside)[0], 4), dtype=tf.float32, name='bbox_inside_weights')
    # only the positive ones have regression targets
    bbox_inside_inds = tf.reshape(tf.where(labels == 1), shape=[-1, ])
    bbox_inside_inds_weights = tf.gather(bbox_inside_weights, bbox_inside_inds) + cfg.train.rpn_bbox_inside_weights
    bbox_inside_inds_expand = tf.expand_dims(bbox_inside_inds, axis=1)
    bbox_inside_weights = tf.tensor_scatter_nd_update(bbox_inside_weights,
                                                      bbox_inside_inds_expand,
                                                      bbox_inside_inds_weights)

    # bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_outside_weights = tf.zeros((tf.shape(inds_inside)[0], 4), dtype=tf.float32, name='bbox_outside_weights')
    if cfg.train.rpn_positive_weight < 0:
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = tf.reduce_sum(tf.cast(labels >= 0, dtype=tf.float32))
        positive_weights = tf.ones((1, 4), dtype=tf.float32) / num_examples
        negative_weights = tf.ones((1, 4), dtype=tf.float32) / num_examples

    else:
        assert ((cfg.train.rpn_positive_weight > 0) & (cfg.train.rpn_positive_weight < 1))
        positive_weights = cfg.train.rpn_positive_weight / tf.reduce_sum(tf.cast(labels == 1, dtype=tf.float32))
        negative_weights = (1.0 - cfg.train.rpn_positive_weight) / tf.reduce_sum(tf.cast(labels == 0, dtype=tf.float32))

    bbox_outside_positive_inds = bbox_inside_inds
    bbox_outside_negative_inds = tf.reshape(tf.where(labels == 0), shape=[-1, ])
    bbox_outside_positive_inds_weights = tf.gather(bbox_outside_weights, bbox_outside_positive_inds) + positive_weights
    bbox_outside_negative_inds_weights = tf.gather(bbox_outside_weights, bbox_outside_negative_inds) + negative_weights
    bbox_outside_positive_inds_expand = tf.expand_dims(bbox_outside_positive_inds, axis=1)
    bbox_outside_negative_inds_expand = tf.expand_dims(bbox_outside_negative_inds, axis=1)
    bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                        bbox_outside_positive_inds_expand,
                                                        bbox_outside_positive_inds_weights)
    bbox_outside_weights = tf.tensor_scatter_nd_update(bbox_outside_weights,
                                                        bbox_outside_negative_inds_expand,
                                                        bbox_outside_negative_inds_weights)

    labels = self.unmap(labels, ori_anchors_num, inds_inside, fill=-1, type='labels')
    bbox_targets = self.unmap(bbox_targets, ori_anchors_num, inds_inside, fill=0, type='bbox_targets')
    bbox_inside_weights = self.unmap(bbox_inside_weights, ori_anchors_num, inds_inside, fill=0, type='bbox_inside_weights')
    bbox_outside_weights = self.unmap(bbox_outside_weights, ori_anchors_num, inds_inside, fill=0, type='bbox_outside_weights')

    rpn_labels = tf.reshape(labels, (1, height, width, A))
    rpn_bbox_targets = tf.reshape(bbox_targets, (1, height, width, A * 4), name='rpn_bbox_targets')
    rpn_bbox_inside_weights = tf.reshape(bbox_inside_weights, (1, height, width, A * 4), name='rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = tf.reshape(bbox_outside_weights, (1, height, width, A * 4),
                                          name='rpn_bbox_outside_weights')
    rpn_labels = tf.cast(rpn_labels, dtype=tf.int32)

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

class proposal_target_layer(tf.keras.layers.Layer): #only in training
  def __init__(self, name = "rpn-data"):
    super().__init__(name)



class rpn(tf.keras.layers.Layer):
  def __init__(self, name = "rpn"):
    super().__init__(name)
    self.rpn_conv = tf.keras.layers.Conv2D(cfg.rpn_chan, kernel_size=(3, 3), padding='same', activation='relu', use_bias=cfg.frcnn.use_bias, kernel_initializer=cfg.frcnn.initializer, kernel_regularizer=cfg.frcnn.regulizer)
    self.proposal_layer = proposal_layer()
  
  def call(self, inputs, img_sz, gt_boxes, training):
    rpn_features = self.rpn_conv(inputs)
    anchors = generate_anchors(rpn_features.shape[0], rpn_features.shape[1])

    proposals, proposal_scores = self.proposal_layer(rpn_features, anchors, img_sz)

    if training == True:
      pass

def get_area(boxes):
  x0, y0, x1, y1 = tf.split(value=boxes, num_or_size_splits=4, axis=1)
  return tf.reshape((x1 - x0 + 1) * (y1 - y0 + 1), (-1, 1))
  
def get_intersection(boxes1, boxes2):
  x0_1, y0_1, x1_1, y1_1 = tf.split(value=boxes1, num_or_size_splits=4, axis=1)
  x0_2, y0_2, x1_2, y1_2 = tf.split(value=boxes2, num_or_size_splits=4, axis=1)
  max_x0 = tf.maximum(x0_1, tf.transpose(x0_2))
  max_y0 = tf.maximum(y0_1, tf.transpose(y0_2))
  min_x1 = tf.minimum(x1_1, tf.transpose(x1_2))
  min_y1 = tf.minimum(y1_1, tf.transpose(y1_2))
  height = tf.maximum(0, min_x1 - max_x0 + 1)
  width = tf.maximum(0, min_y1 - max_y0 + 1)
  return height * width

def get_overlap(anchors, gt_boxes):
  anchor_area = get_area(anchors)
  gt_boxes_area = get_area(gt_boxes)
  intersection = get_intersection(anchors, gt_boxes)
  union = anchor_area + tf.reshape(gt_boxes_area, (1, -1)) - intersection
  return intersection / union


def bbox_transform_tf(anchors, gt_boxes):
  anchor_h = anchors[:, 2] - anchors[:, 0] + 1.0
  anchor_w = anchors[:, 3] - anchors[:, 1] + 1.0
  anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_h
  anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_w

  box_h = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
  box_w = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
  gt_ctr_x = gt_boxes[:, 0] + 0.5 * box_h
  gt_ctr_y = gt_boxes[:, 1] + 0.5 * box_w

  targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_h
  targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_w
  targets_dw = tf.math.log(box_h / anchor_h)
  targets_dh = tf.math.log(box_w / anchor_w)

  targets = tf.stack([targets_dx, targets_dy, targets_dw, targets_dh], axis=1)
  return targets

def bbox_transform_inv(anchors, pred):
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
  y0 = y_pred - 0.5 * w_pred
  y1 = y_pred + 0.5 * h_pred

  return tf.stack([x0, y0, x1, y1], axis = -1)