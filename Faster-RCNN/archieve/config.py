from easydict import EasyDict as edict
import numpy as np
import tensorflow as tf

# pascal voc 2012
pascal_voc_2012 = edict()
pascal_voc_2012.classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                    'sheep', 'sofa', 'train', 'tvmonitor']
pascal_voc_2012.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
pascal_voc_2012.max_size = 1000
pascal_voc_2012.min_size = 600

# vgg16
vgg16 = edict()
vgg16.feature_stride = 16 # when input size = 224, output = 14, strive = 224/14

# anchor
anchors = edict()
anchors.ratios = np.array([0.5, 1, 2])
anchors.scales = np.array([8, 16, 32])
anchors.num = len(anchors.ratios) * len(anchors.scales) # 1 anchor for each combination

# region proposal network
rpn = edict()
rpn.rpn_chan = 512

# arch
frcnn = edict()
frcnn.use_bias = False
frcnn.regulizer = None
frcnn.initializer = tf.keras.initializers.RandomNormal(stddev=0.01)
frcnn.backbone = tf.keras.applications.vgg16.VGG16(False, weights = 'imagenet')

#train
train = edict()
train.nms.max_output_size = 2000
train.nms.iou_threshold = 0.7
train.rpn_min_size = 16
train.rpn_negative_threshold = 0.3
train.rpn_positive_threshold = 0.7
train.batch_size = 256
train.positive_frac = 0.5
train.rpn_bbox_inside_weights=(1.0, 1.0, 1.0, 1.0)
train.rpn_positive_weight = 1.0

# test
test = edict()
test.nms.max_output_size = 300
test.nms.iou_threshold = 0.7
test.rpn_min_size = 16

# overall
cfg = edict()
cfg.db = pascal_voc_2012
cfg.backbone = vgg16
cfg.anchors = anchors
cfg.frcnn = frcnn
cfg.train = train
cfg.test = test
