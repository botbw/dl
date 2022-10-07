
import os

import numpy as np

import xmltodict
from PIL import Image

# pascal
class pascal_voc_detection:
  def __init__(self, path):
    self.path = path
    self.annotation_dir = os.path.join(path, "Annotations")
    self.layout_dir = os.path.join(path, "ImageSets", "Main")
    self.img_dir = os.path.join(path, "JPEGImages")
    self.trainval_list_file = os.path.join(self.layout_dir, 'trainval.txt')
    self.test_list_file = os.path.join(self.layout_dir, 'test.txt')
    self.classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
                        'sheep', 'sofa', 'train', 'tv/monitor']
    self.c2i = dict()
    for i in range(len(self.classes)):
      dict[self.classes[i]] = i

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
        id = self.c2i[box['name']]
        xmin = int(float(box['bndbox']['xmin']))
        ymin = int(float(box['bndbox']['ymin']))
        xmax = int(float(box['bndbox']['xmax']))
        ymax = int(float(box['bndbox']['ymax']))
        _boxes.append([xmin, ymin, xmax, ymax, id])
      boxes.append(_boxes)
    return boxes
        
  def load(self):
    train_imgs, train_annos = self.load_using_list_file(self.trainval_list_file)
    test_imgs, test_annos = self.load_using_list_file(self.test_list_file)
    return train_imgs, train_annos



