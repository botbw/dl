
from genericpath import isdir
import os

import numpy as np

import tensorflow as tf

import pickle
import xmltodict
from PIL import Image

# pascal
class pascal_voc_detection:
  def __init__(self, path):
    self.path = path
    self.annotation_dir = os.path.join(path, "Annotations")
    self.layout_dir = os.path.join(path, "ImageSets", "Main")
    self.img_dir = os.path.join(path, "JPEGImages")
    self.trainval_list_file = os.path.join(self.layout_dir, 'train.txt')
    self.test_list_file = os.path.join(self.layout_dir, 'val.txt')


  def load_using_list_file(self, list_file):
    img_paths = []
    anno_paths = []
    cnt = 0
    with open(list_file) as f:
      for line in f:
        cnt += 1
        filename = line.strip()
        jpg_file = os.path.join(self.img_dir, f"{filename}.jpg")
        jpeg_file = os.path.join(self.img_dir, f"{filename}.jpeg")
        annotation_file = os.path.join(self.annotation_dir, f"{filename}.xml")
        if os.path.isfile(annotation_file): # annotation should exist
          anno_paths.append(annotation_file)
          if os.path.isfile(jpg_file): # suffix is jpg
            img_paths.append(jpg_file)
          elif os.path.isfile(jpeg_file):
            img_paths.append(jpeg_file)
          else:
            print(f"{filename} img not exists")
        else :
          print(f"{filename} annotation not exists")
        if cnt == 10:
          break
      imgs = []
      annos = []
      for path in img_paths:
        imgs.append(np.asarray(Image.open(path)))
      for path in anno_paths:
        with open(path) as f:
          annos.append(xmltodict.parse(f.read())['annotation'])
    return imgs, self.extract_from_annos(annos)

  def extract_from_annos(self, annos):
    boxes = []
    for i in range(len(annos)):
      obj = annos[i]['object']
      _boxes = []
      if type(obj) == list:
        for box in obj:
          id = self.c2i[box['name']]
          xmin = int(float(box['bndbox']['xmin']))
          ymin = int(float(box['bndbox']['ymin']))
          xmax = int(float(box['bndbox']['xmax']))
          ymax = int(float(box['bndbox']['ymax']))
          _boxes.append([xmin, ymin, xmax, ymax, id])
      else:
        box = annos[i]['object']
        id = box['name']
        xmin = int(float(box['bndbox']['xmin']))
        ymin = int(float(box['bndbox']['ymin']))
        xmax = int(float(box['bndbox']['xmax']))
        ymax = int(float(box['bndbox']['ymax']))
        _boxes.append([xmin, ymin, xmax, ymax, id])
      boxes.append(_boxes)
    assert(len(boxes) == len(annos))
    return boxes
        
  def load(self, cache_path = os.path.join("cache")):
    try:
      with open(os.path.join(cache_path, "pascal_voc_2012_train_imgs.pkl"), "rb") as f:
        train_imgs = pickle.load(f)
      with open(os.path.join(cache_path, "pascal_voc_2012_train_annos.pkl"), "rb") as f:
        train_annos = pickle.load(f)
      with open(os.path.join(cache_path, "pascal_voc_2012_test_imgs.pkl"), "rb") as f:
        test_imgs = pickle.load(f)
      with open(os.path.join(cache_path, "pascal_voc_2012_test_annos.pkl"), "rb") as f:
        test_annos = pickle.load(f)
      print("loaded using cache file")
    except:
      print("cache not found, load and cache")
      if not os.path.exists(cache_path):
        os.mkdir(cache_path)
      train_imgs, train_annos = self.load_using_list_file(self.trainval_list_file)
      test_imgs, test_annos = self.load_using_list_file(self.test_list_file)
      
      with open(os.path.join(cache_path ,"pascal_voc_2012_train_imgs.pkl"), "wb") as f:
        pickle.dump(train_imgs, f)
      with open(os.path.join(cache_path, "pascal_voc_2012_train_annos.pkl"), "wb") as f:
        pickle.dump(train_annos, f)
      with open(os.path.join(cache_path, "pascal_voc_2012_test_imgs.pkl"), "wb") as f:
        pickle.dump(test_imgs, f)
      with open(os.path.join(cache_path, "pascal_voc_2012_test_annos.pkl"), "wb") as f:
        pickle.dump(test_annos, f)
    return (train_imgs, train_annos), (test_imgs, test_annos)
