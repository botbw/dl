import os
import utils.datasets
import numpy as np
import tensorflow as tf
from config import cfg


def preprocessing():
  db = utils.datasets.pascal_voc_detection(os.path.normpath("/home/FYP/wang1570/datasets/pascal-voc-2012/VOCdevkit/VOC2012"))
  c2i = dict()
  classes = cfg.db.classes
  for i in range(len(classes)):
    c2i[classes[i]] = i
  (train_imgs, train_annos), (test_imgs, test_annos) = db.load()
  train_imgs, train_annos = resize(train_imgs, train_annos)
  test_imgs, test_annos = resize(test_imgs, test_annos)
  return (train_imgs, train_annos), (test_imgs, test_annos)

def resize(imgs, annos):
  scaled_imgs = []
  scaled_annos = []
  for img, anno_list in zip(imgs, annos):
    scaled_img, scaled_anno = __resize(img, anno_list)
    scaled_imgs.append(scaled_img)
    scaled_annos.append(scaled_anno)
  return np.array(scaled_imgs, dtype = 'float'), np.array(scaled_annos)

def __resize(img, anno_list):
  img -= cfg.db.pixel_means
  shape = img.shape[0:2]
  shorter_side = min(shape)
  longer_side = max(shape)
  scaling = float(cfg.db.min_size)/float(shorter_side)
  if np.round(scaling * longer_side) > cfg.db.max_size:
    scaling = float(cfg.db.max_size) / float(longer_side)
  scaled_img = tf.image.resize(img, np.asarray(shape, dtype = 'float') * scaling)
  scaled_anno_list = []
  for anno in anno_list:
    scaled_anno = np.asarray(anno) * scaling
    scaled_anno_list.append(scaled_anno)
  return scaled_img, scaled_anno_list
  