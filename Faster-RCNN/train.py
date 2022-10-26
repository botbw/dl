from faster_rcnn import FasterRCNN
frcnn = FasterRCNN(rpn_positive_overlap=0.7,
                       classes=['__background__','bird', 'cat', 'cow', 'dog', 'horse', 'sheep','aeroplane',
                                'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair',
                                'diningtable', 'pottedplant', 'sofa', 'tvmonitor','person'])
frcnn.train(epochs=100, data_root_path='/home/FYP/wang1570/datasets/pascal-voc-2012/VOCdevkit/VOC2012', log_dir='./logs', save_path='./')