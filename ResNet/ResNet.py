from tensorflow.keras import layers, Sequential
import tensorflow.keras as keras

class ResBlk(layers.Layer):
    expansion = 1
    def __init__(self, out_chan, strides = 1, downsample = None, **kwargs):
        super().__init__()
        
        self.conv1 = layers.Conv2D(out_chan, padding='same', kernel_size=3, strides=strides)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(out_chan, padding='same', kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()
        
        self.downsample = downsample
        
        self.relu2 = layers.ReLU()

    
    def call(self, X, training = False, **kwargs):
        y = X
        
        y = self.conv1(y)
        y = self.bn1(y, training)
        y = self.relu1(y)

        
        y = self.conv2(y)
        y = self.bn2(y, training)
        
        h_x = X

        if self.downsample != None:
            h_x = self.downsample(h_x)
  
        y += h_x
        y = self.relu2(y)

        return y

class BottleneckBlk(layers.Layer):
    expansion = 4
    def __init__(self, out_chan, strides = 1, downsample = None, **kwargs):
        super().__init__()
   
        self.conv1 = layers.Conv2D(out_chan, padding='same', kernel_size=1, strides=strides)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(out_chan, padding='same', kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(out_chan * self.expansion, padding='same', kernel_size=1, strides=1)
        self.bn3 = layers.BatchNormalization()

        self.downsample = downsample

        self.relu3 = layers.ReLU()

    def call(self, X, training = False, **kwargs):
        y = X
        
        y = self.conv1(y)
        y = self.bn1(y, training)
        y = self.relu1(y)
        
        y = self.conv2(y)
        y = self.bn2(y, training)
        y = self.relu2(y)
        
        y = self.conv3(y)
        y = self.bn3(y, training)

        h_x = X
        if self.downsample != None:
            h_x = self.downsample(h_x)
        y += h_x
        y = self.relu3(y)

        return y

class ResNet(keras.Model):
  def __init__(self, blkType, blk_num_list, **kwargs):
    if len(blk_num_list) != 4:
      raise Exception("The lenght blk_num_list should be 4")

    super().__init__(**kwargs)

    # following paper name convention
    conv1 = keras.Sequential(name = "conv1")
    conv1.add(layers.Conv2D(64, kernel_size = 7, strides = 2, padding = "same"))
    conv1.add(layers.BatchNormalization())
    conv1.add(layers.ReLU())

    self.conv1 = conv1

    self.conv2_pool = layers.MaxPool2D(3, strides = 2, padding = "same")
    self.conv2_x = self.build_stage(blkType, True, 64, blk_num_list[0], strides = 1, name = 'conv2_x')

    self.conv3_x = self.build_stage(blkType, False, 128, blk_num_list[1], strides = 2, name = 'conv3_x')

    self.conv4_x = self.build_stage(blkType, False, 256, blk_num_list[2], strides = 2, name = 'conv4_x')

    self.conv5_x = self.build_stage(blkType, False, 512, blk_num_list[3], strides = 2, name = 'conv5_x')
  
  def build_stage(self, blkType, isFirstStage, out_chan, blk_num, strides = 1, name = None):
    downsample = None
    if strides != 1 or isFirstStage == False or blkType == BottleneckBlk:
      downsample = keras.Sequential([
          layers.Conv2D(out_chan * blkType.expansion, kernel_size=1, strides=strides),
          layers.BatchNormalization()
      ])

    stage = keras.Sequential(name = name)
    stage.add(blkType(out_chan, downsample = downsample, strides = strides))

    for i in range(1, blk_num):
      stage.add(blkType(out_chan, strides = 1))
    
    return stage

  def call(self, X, training = False, **kwargs):
    y = self.conv1(X)
    y = self.conv2_pool(y)
    y = self.conv2_x(y)
    y = self.conv3_x(y)
    y = self.conv4_x(y)
    y = self.conv5_x(y)

    return y
  
def resnet_34():
  blkType = ResBlk
  blk_num_list = [3, 4, 6, 3]
  return ResNet(blkType, blk_num_list)

def resnet_18():
  blkType = ResBlk
  blk_num_list = [2, 2, 2, 2]
  return ResNet(blkType, blk_num_list)

def resnet_50():
  blkType = BottleneckBlk
  blk_num_list = [3, 4, 6, 3]
  return ResNet(blkType, blk_num_list)

def resnet_101():
  blkType = BottleneckBlk
  blk_num_list = [3, 4, 23, 3]
  return ResNet(blkType, blk_num_list)

def resnet_152():
  blkType = BottleneckBlk
  blk_num_list = [3, 8, 36, 3]
  return ResNet(blkType, blk_num_list)
